import glob
import random
from collections import Counter

import h5py
import imblearn
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import scipy
import seaborn as sns
import sklearn
import tensorflow as tf
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             make_scorer, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

IMAGE_SIZE = (224,224,3)
NUM_FOLDS = 5

def load_devices(use_gpu=0):
  if use_gpu:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
  else: tf.config.set_visible_devices([], "GPU")

def create_dataset_files(labels_root="FL_imaging/FL_labels.xlsx", data_root="FL_imaging", train_destination="FL_compressed/train_data.hdf5", holdout_destination="FL_compressed/holdout_data.hdf5", image_size=IMAGE_SIZE, crop_fraction=0.7, num_folds=NUM_FOLDS):
  labels_df = pd.read_excel(labels_root, sheet_name=None)["Sheet1"]
  labels_df = labels_df[labels_df.columns[:2]].sort_values(by="Pt ID")
  labels_df = labels_df[labels_df['Fat%'].notna()]
  
  def _compute_labels(percent):
    if percent < 5: return 0.
    elif percent <= 17.4: return 1.
    else: return 2.
  
  labels_df["Labels"] = labels_df["Fat%"].apply(_compute_labels)
  
  patient_labels = np.array(labels_df["Labels"])
  num_patients = len(patient_labels)
  
  patients = sorted(glob.glob(f"{data_root}/*/*/"))
  jpeg_paths = [glob.glob(f"{patient}Static JPG/*.jpeg") for patient in patients][:num_patients]
  dicom_paths = [glob.glob(f"{patient}Cine DICOM/*.dcm") for patient in patients][:num_patients]
  patient_paths = np.array([j+d for j,d in zip(jpeg_paths, dicom_paths)], dtype=object)
  
  indices_with_no_data = [i for i, paths in enumerate(patient_paths) if not len(paths)]
  patient_labels = np.delete(patient_labels, indices_with_no_data, axis=0)
  patient_paths = np.delete(patient_paths, indices_with_no_data, axis=0)
  
  patient_labels, patient_paths = sklearn.utils.shuffle(patient_labels, patient_paths)

  num_images = 0
  image_labels = []
  image_groups = []
  print("--------------- Metadata Preloading ---------------")
  for i, paths in enumerate(patient_paths):
    print(f"Preloaded Patients: {i}")
    for _, path in enumerate(paths):
      try:
        dicom = pydicom.read_file(path, force=True)
        no_of_times = dicom.NumberOfFrames
      except: no_of_times = 1
      finally:
        for _ in range(no_of_times):
          image_labels.append(patient_labels[i])
          image_groups.append(i+1)
          num_images += 1
  # print(f"Preloaded Patients: {len(patient_paths)}")
  
  holdout_kfold = StratifiedGroupKFold(n_splits=num_folds, shuffle=True)
  holdout_kfold_splits = holdout_kfold.split(np.zeros(num_images), image_labels, image_groups)
  
  _, test_indices = list(holdout_kfold_splits)[random.randrange(0,num_folds)]
  holdout_indices = sorted(test_indices.tolist())
  
  num_train_images = 0
  num_holdout_images = 0
  
  train_patient_labels = []
  holdout_patient_labels = []
  
  train_image_labels = []
  holdout_image_labels = []
  
  train_image_groups = []
  holdout_image_groups = []
  
  train_unique_patients = []
  holdout_unique_patients = []
  
  print("--------------- Image Population ---------------")
  for i, paths in enumerate(patient_paths):
    print(f"Populated Patients: {i}")
    for _, path in enumerate(paths):
      try:
        dicom = pydicom.read_file(path, force=True)
        array = dicom.pixel_array.astype(np.float32)
        
        no_of_times = dicom.NumberOfFrames
        images = [np.stack(
                    (tf.image.resize(
                      tf.image.central_crop(
                        array[j] if len(array.shape) == 4 else np.pad(array[j][:,:,np.newaxis],((0,0),(0,0),(0,2)), constant_values=128.),
                          crop_fraction), image_size[:-1])[:,:,0],)*image_size[-1], axis=-1) for j in range(no_of_times)]
      except:
        no_of_times = 1
        images = [tf.image.resize(
                    tf.image.central_crop(
                      tf.keras.utils.img_to_array(
                        tf.keras.utils.load_img(path, color_mode="rgb", interpolation="bilinear")),
                          crop_fraction), image_size[:-1])]
      finally:
        for k in range(no_of_times):
          if (num_train_images+num_holdout_images) in holdout_indices:
            with h5py.File(holdout_destination,"a") as holdout_file: holdout_file.create_dataset(str(num_holdout_images), data=images[k])
            
            num_holdout_images += 1
            holdout_image_labels.append(patient_labels[i])
            
            if i not in holdout_unique_patients: 
              holdout_unique_patients.append(i)
              holdout_patient_labels.append(patient_labels[i])
            
            holdout_image_groups.append(len(holdout_unique_patients))
          else:
            with h5py.File(train_destination,"a") as train_file: train_file.create_dataset(str(num_train_images), data=images[k])
            
            num_train_images += 1
            train_image_labels.append(patient_labels[i])
            
            if i not in train_unique_patients: 
              train_unique_patients.append(i)
              train_patient_labels.append(patient_labels[i])
            
            train_image_groups.append(len(train_unique_patients))
  
  train_patient_labels, holdout_patient_labels = np.array(train_patient_labels), np.array(holdout_patient_labels)
  train_image_labels, holdout_image_labels = np.array(train_image_labels), np.array(holdout_image_labels)
  train_image_groups, holdout_image_groups = np.array(train_image_groups), np.array(holdout_image_groups)
  
  with h5py.File(train_destination,"a") as train_file, h5py.File(holdout_destination,"a") as holdout_file:
    train_file.create_dataset("N", data=num_train_images)
    holdout_file.create_dataset("N", data=num_holdout_images)
    
    train_file.create_dataset("P", data=train_patient_labels)
    holdout_file.create_dataset("P", data=holdout_patient_labels)
    
    train_file.create_dataset("L", data=train_image_labels)
    holdout_file.create_dataset("L", data=holdout_image_labels)
    
    train_file.create_dataset("G", data=train_image_groups)
    holdout_file.create_dataset("G", data=holdout_image_groups)

def compute_class_weights(labels):
      count = Counter([int(label) for label in labels])
      factor = sum(count.values()) / len(count)
      for key, val in count.most_common():
        count[key] = (1 / val) * factor
      return count

def compute_scale_pos_weight(labels):
  count = [val for _, val in Counter([int(label) for label in labels]).most_common()]
  return count[0]/count[-1]

def create_frame_level_model(input_shape=IMAGE_SIZE):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(input_shape))
    
    pretrained_model = tf.keras.applications.MobileNetV3Small(include_top=False,
                                                              input_shape=input_shape,
                                                              weights="imagenet"
                                                              )
    pretrained_model.trainable = False
    model.add(pretrained_model)
    model.add(tf.keras.layers.GroupNormalization())
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return model

def plot_cv_results(val_labels, val_probs, thresholds=[p/20 for p in range(0,20)]):
  def plot_curves_together(title, xs, ys, colors, labels, xlabel, ylabel, xlim=[0,1], ylim=[0,1], lloc="lower right"):
    plt.title(title)
    for i in range(len(xs)): plt.plot(xs[i], ys[i], color=colors[i], label=labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc=lloc)
    plt.show()
  
  def plot_separate_curves(titles, xs, ys, colors, labels, xlabels, ylabels, xlim=[0,1], ylim=[0,1], lloc="lower right"):
    for i in range(len(xs)):
      plt.title(titles[i])
      plt.plot(xs[i], ys[i], color=colors[i], label=labels[i])
      plt.xlabel(xlabels[i])
      plt.ylabel(ylabels[i])
      plt.xlim(xlim)
      plt.ylim(ylim)
      plt.legend(loc=lloc)
      plt.show()
  
  def plot_confusion_matrix(labels, predictions):
    confusion_matrix = sklearn.metrics.confusion_matrix(labels, predictions)
    display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix)
    display.plot()
    plt.show()
  
  fpr, tpr, _ = sklearn.metrics.roc_curve(val_labels, val_probs)
  roc_auc_score = sklearn.metrics.auc(fpr, tpr)
  
  plot_curves_together("Receiver Operating Characteristic", [fpr, [0,1]], [tpr, [0,1]], ["b", "r"], [f"AUC={roc_auc_score}", ""], "False Positive Rate", "True Positive Rate")
  
  accuracy_scores = []
  precision_scores = []
  recall_scores = []
  f1_scores = []
  sensitivity_scores = []
  specificity_scores = []
  
  for t in thresholds:
    val_predictions = [float(prob>t) for prob in val_probs]
    accuracy = sklearn.metrics.accuracy_score(val_labels, val_predictions)
    precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(val_labels, val_predictions, average="binary")
    sensitivity, specificity, _ = imblearn.metrics.sensitivity_specificity_support(val_labels, val_predictions, average="binary")
    
    if t==0.5: plot_confusion_matrix(val_labels, val_predictions)
    
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1_score)
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)
  
  scores = [accuracy_scores, precision_scores, recall_scores, f1_scores, sensitivity_scores, specificity_scores]
  xdata = [thresholds for _ in range(len(scores))]
  colors = ["r" for _ in range(len(scores))]
  labels = [f"{score[thresholds.index(0.5)]} at t=0.5" for score in scores]
  xlabels = ["Thresholds"] * len(scores)
  ylabels = ["Accuracy", "Precision", "Recall", "F1-Score", "Sensitivity", "Specificity"]
  titles = [f"{ylabels[i]} vs. {xlabels[i]}" for i in range(len(xlabels))]
  
  plot_separate_curves(titles, xdata, scores, colors, labels, xlabels, ylabels)

class DataSequence(tf.keras.utils.Sequence):
    def __init__(self, train_data_root="FL_compressed/train_data.hdf5", data_shape=IMAGE_SIZE, binary_split=-1):
      super().__init__()
      self.train_data_root = train_data_root
      self.data_shape = data_shape
      self.binary_split = binary_split
      self._collect_metadata()
    
    def _collect_metadata(self):
      with h5py.File(self.train_data_root, "r") as data_file:
        self.num_images = data_file["N"][...]
        self.patient_labels = (data_file["P"][...]>self.binary_split).astype(np.float32) if self.binary_split>-1 else (data_file["P"][...]).astype(np.float32)
        self.image_labels = (data_file["L"][...]>self.binary_split).astype(np.float32) if self.binary_split>-1 else (data_file["L"][...]).astype(np.float32)
        self.image_groups = data_file["G"][...]
    
    def __len__(self): return self.num_images
    
    def _generate_items(self, indices, data=0, labels=0):
      for i in indices:
        if labels: yield self.image_labels[i]
        else:
          with h5py.File(self.train_data_root, "r") as data_file: image = data_file[str(i)][...]
          yield image[None, ...] if data else (image[None, ...], self.image_labels[i][None, None, ...])
    
    def generate_data(self, indices): return self._generate_items(indices, data=1)
    
    def list_labels(self, indices): return [label for label in self._generate_items(indices, labels=1)]
    
    def __getitem__(self, indices): return tf.data.Dataset.from_generator(lambda: self._generate_items(indices), output_signature=(tf.TensorSpec(shape=(None,*self.data_shape), dtype=tf.float32), tf.TensorSpec(shape=(None,1), dtype=tf.float32))).repeat()
    
    def sort_indices_into_groups(self, indices):
      groups = []
      grouped_indices = []
      
      for idx in indices:
        current_group = self.image_groups[idx]
        if current_group not in groups:
          groups.append(current_group)
          grouped_indices.append([idx])
        else:
          place_to_insert = groups.index(current_group)
          grouped_indices[place_to_insert].append(idx)
      
      return groups, grouped_indices
    
    def list_patient_labels(self, indices):
      groups, _ = self.sort_indices_into_groups(indices)
      scores = [self.patient_labels[group-1] for group in groups]
      
      return np.array(scores)
    
    def compute_patient_level_features(self, frame_level_probabilities, indices):
      _, grouped_indices = self.sort_indices_into_groups(indices)
      values = []
      for grouped_set in grouped_indices:
        grouped_set = [np.where(indices==idx)[0][0] for idx in grouped_set]
        probs_per_set = frame_level_probabilities[grouped_set]
        values.append([np.nanmin(probs_per_set), np.nanmax(probs_per_set), np.ptp(probs_per_set), np.mean(probs_per_set), np.var(probs_per_set), np.std(probs_per_set, ddof=1), scipy.stats.gmean(probs_per_set)[0], scipy.stats.gstd(probs_per_set)[0], scipy.stats.hmean(probs_per_set)[0], scipy.stats.sem(probs_per_set)[0], scipy.stats.variation(probs_per_set)[0], np.median(probs_per_set), stats.iqr(probs_per_set), scipy.stats.mode(probs_per_set).mode[0], scipy.stats.skew(probs_per_set)[0], scipy.stats.kurtosis(probs_per_set)[0], scipy.stats.entropy(probs_per_set)[0], scipy.stats.differential_entropy(probs_per_set)[0], scipy.stats.median_abs_deviation(probs_per_set)[0]])
      
      return np.array(values)

def models_cv(num_folds=NUM_FOLDS-1, epochs_per_fold=2, batch_size_per_fold=1, frame_level_exists=0):
  for split in range(0,2):
    data_sequence = DataSequence(binary_split=split)
    
    full_data_indices = np.arange(len(data_sequence))
    shuffled_full_data_indices = np.random.permutation(len(data_sequence))
    
    features_file = "new_models/cv_features/second_class_patient_features.npy" if split else "new_models/cv_features/first_class_patient_features.npy"
    
    if frame_level_exists==2:
      with open(features_file, "rb") as ff: patient_level_features = np.load(ff)
    else:
      if frame_level_exists==0:
        print(f'--------------- Training Frame Level Model ---------------')
        
        model = create_frame_level_model()
      
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        print(next(data_sequence.generate_data([0])).shape)
        model.fit(data_sequence[shuffled_full_data_indices],
                  steps_per_epoch=len(shuffled_full_data_indices),
                  epochs=epochs_per_fold,
                  batch_size=batch_size_per_fold,
                  class_weight=compute_class_weights(data_sequence.list_labels(shuffled_full_data_indices)),
                  )
        
        model.save("new_models/frame_level/second_class_cnn.h5") if split else model.save("new_models/frame_level/first_class_cnn.h5")
      elif frame_level_exists==1:
        model = tf.keras.models.load_model("new_models/frame_level/second_class_cnn.h5") if split else tf.keras.models.load_model("new_models/frame_level/first_class_cnn.h5")
      
      frame_level_features = model.predict(data_sequence.generate_data(full_data_indices))
      patient_level_features = data_sequence.compute_patient_level_features(frame_level_features, full_data_indices)
      
      with open(features_file, "wb") as ff: np.save(ff, patient_level_features)
      
    train_patient_labels = data_sequence.list_patient_labels(full_data_indices)
    
    cv_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
    cv_kfold_splits = list(cv_kfold.split(patient_level_features, train_patient_labels))
    for train_indices, val_indices in cv_kfold_splits: random.shuffle(train_indices); random.shuffle(val_indices)
    
    cv_param_grid_boost = {
      "max_depth": [None],
      "learning_rate": [0.01],
      "max_iter": [1000],
      "max_leaf_nodes": [7, 15, 31, 63],
      "min_samples_leaf": [20, 30, 40],
      "l2_regularization": [0., 0.25, 0.5, 0.75, 1.],
      "class_weight": ["balanced", None],
      "scoring": ["sensitivity"],
      "validation_fraction": [None]
    }
    
    cv_scoring = {
      "roc_auc": make_scorer(roc_auc_score),
      "balanced_accuracy": make_scorer(balanced_accuracy_score),
      "accuracy": make_scorer(accuracy_score),
      "sensitivity": make_scorer(recall_score),
      "specificity": make_scorer(recall_score, pos_label=0),
      "precision": make_scorer(precision_score),
      "f1_score": make_scorer(f1_score)
    }
    
    cv_search_boost = GridSearchCV(estimator=HistGradientBoostingClassifier(), param_grid=cv_param_grid_boost, scoring=cv_scoring, refit="sensitivity", cv=cv_kfold_splits, n_jobs=-1, verbose=3, return_train_score=True)
    cv_search_boost.fit(patient_level_features, train_patient_labels)
    
    cv_results = pd.DataFrame(cv_search_boost.cv_results_)
    cv_results.to_excel("new_models/cv_results/second_class_cv_results.xlsx") if split else cv_results.to_excel("new_models/cv_results/first_class_cv_results.xlsx")
    
    patient_level_classifier = cv_search_boost.best_estimator_
    joblib.dump(patient_level_classifier, "new_models/patient_boost/second_class_boost.joblib") if split else joblib.dump(patient_level_classifier, "new_models/patient_boost/first_class_boost.joblib")

def evaluate_models():
  data = DataSequence(train_data_root="FL_compressed/holdout_data.hdf5")
  indexes = np.array(list(range(0,len(data))))
  
  frame_model_one = tf.keras.models.load_model("new_models/frame_level/first_class_cnn.h5")
  frame_model_two = tf.keras.models.load_model("new_models/frame_level/second_class_cnn.h5")
    
  frame_probs_one = frame_model_one.predict(data.generate_data(indexes))
  patient_features_one = data.compute_patient_level_features(frame_probs_one, indexes)
    
  frame_probs_two = frame_model_two.predict(data.generate_data(indexes))
  patient_features_two = data.compute_patient_level_features(frame_probs_two, indexes)
  
  patient_svm_one = joblib.load("new_models/patient_boost/first_class_boost.joblib")
  patient_svm_two = joblib.load("new_models/patient_boost/second_class_boost.joblib")
  
  patient_probs_one = np.array([prob[1] for prob in patient_svm_one.predict_proba(patient_features_one)])
  patient_probs_two = np.array([prob[1] for prob in patient_svm_two.predict_proba(patient_features_two)])

  true_patient_labels = data.list_patient_labels(indexes)
  first_binary_labels = np.where(true_patient_labels>0.,1.,0.)
  second_binary_labels = np.where(true_patient_labels>1.,1.,0.)
  
  fpr, tpr, _ = sklearn.metrics.roc_curve(first_binary_labels, patient_probs_one, drop_intermediate=False)
  roc_auc_score = sklearn.metrics.auc(fpr, tpr)

  plt.plot(fpr, tpr, color="blue", label=f"Model 1 (AUC: {round(roc_auc_score,3)})")
  
  fpr, tpr, _ = sklearn.metrics.roc_curve(second_binary_labels, patient_probs_two, drop_intermediate=False)
  roc_auc_score = sklearn.metrics.auc(fpr, tpr)
  
  plt.plot(fpr, tpr, color="green", label=f"Model 2 (AUC: {round(roc_auc_score,3)})")
  plt.plot([0,1], [0,1], color="red", linestyle="dashed")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.legend(loc = "lower right")
  plt.xlim([0,1])
  plt.ylim([0,1])
  plt.show()
  
  thresholds = [p/20 for p in range(1,20)]
  
  sens_1 = []
  spec_1 = []
  sens_2 = []
  spec_2 = []
  for t in thresholds:
    print(f"-------THRESHOLD {t}-------")
    first_binary_labels = np.where(true_patient_labels>0.,1.,0.)
    first_binary_pred = np.where(patient_probs_one>t,1.,0.)
    
    second_binary_labels = np.where(true_patient_labels>1.,1.,0.)
    second_binary_pred = np.where(patient_probs_two>t,1.,0.)
    
    first_sens, first_spec, _ = imblearn.metrics.sensitivity_specificity_support(first_binary_labels, first_binary_pred, average="binary")
    second_sens, second_spec, _ = imblearn.metrics.sensitivity_specificity_support(second_binary_labels, second_binary_pred, average="binary")
    
    sens_1.append(first_sens)
    spec_1.append(first_spec)
    
    sens_2.append(second_sens)
    spec_2.append(second_spec)
  
  plt.title("Model 1 Sensitivity")
  plt.plot(thresholds, sens_1, label = f"{sens_1[thresholds.index(0.5)]} at t=0.5")
  plt.legend(loc = "lower right")
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.show()
  
  plt.title("Model 1 Specificity")
  plt.plot(thresholds, spec_1, label = f"{spec_1[thresholds.index(0.5)]} at t=0.5")
  plt.legend(loc = "lower right")
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.show()
  
  plt.title("Model 2 Sensitivity")
  plt.plot(thresholds, sens_2, label = f"{sens_2[thresholds.index(0.5)]} at t=0.5")
  plt.legend(loc = "lower right")
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.show()
  
  plt.title("Model 2 Specificity")
  plt.plot(thresholds, spec_2, label = f"{spec_2[thresholds.index(0.5)]} at t=0.5")
  plt.legend(loc = "lower right")
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.show()
  
  thresholds = [p/20 for p in range(1,20)]
  accuracies = []
  f1_grade = [[], [], []]
  sensitivity_grade = [[], [], []]
  specificity_grade = [[], [], []]
  avgf1 = []
  for t in thresholds:
    first_scores = np.where(patient_probs_one>t,1.,0.)
    second_scores = np.where(patient_probs_two>t,1.,0.)
    
    predicted = []
    for i in range(len(first_scores)):
      if second_scores[i] == 1.: predicted.append(2.)
      elif first_scores[i] == 1.: predicted.append(1.)
      else: predicted.append(0.)
      
    """
    if t==0.5:
      confusion_matrix = sklearn.metrics.confusion_matrix(true_patient_labels, np.array(predicted))
      norm_confusion_matrix = np.transpose(np.transpose(confusion_matrix) / confusion_matrix.astype(np.float64).sum(axis=1))
      
      labels=[["27\n(0.794)", "6\n(0.176)", "1\n(0.029)"], ["4\n(0.143)", "23\n(0.821)", "1\n(0.036)"], ["0\n(0.000)", "3\n(0.176)", "14\n(0.824)"]]
      hm = sns.heatmap(norm_confusion_matrix, annot=labels, fmt="", cmap="Blues")
      hm.set(xlabel="Predicted label", ylabel="True label")
      plt.yticks(rotation=0)
      plt.show()
    """

    accuracies.append(sklearn.metrics.accuracy_score(true_patient_labels, predicted))
    f1_scores = sklearn.metrics.f1_score(true_patient_labels, predicted, average=None, labels=[0,1,2])
    recall_pos, recall_neg, _ = imblearn.metrics.sensitivity_specificity_support(true_patient_labels, predicted, average=None, labels=[0,1,2])
    for i in range(3):
      f1_grade[i].append(f1_scores[i])
      sensitivity_grade[i].append(recall_pos[i])
      specificity_grade[i].append(recall_neg[i])
    
    avgf1.append(sum(f1_scores.tolist())/len(f1_scores.tolist()))
      
    print(f"--------Threshold {t}---------")
    print(f1_scores)
    print(true_patient_labels)
    print(predicted)

  plt.title(f"Threshold vs. Accuracy")
  plt.plot(thresholds, accuracies, "r", label = f"{accuracies[thresholds.index(0.5)]} at t=0.5")
  plt.legend(loc = "lower right")
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel(f"Accuracy")
  plt.xlabel("Threshold")
  plt.show()
  
  for i in range(3):
    plt.title(f"Threshold vs. Grade {i} Sensitivity")
    plt.plot(thresholds, sensitivity_grade[i], "r", label = f"{sensitivity_grade[i][thresholds.index(0.5)]} at: t=0.5")
    plt.legend(loc = "lower right")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel(f"Grade {i} Sensitivity")
    plt.xlabel("Threshold")
    plt.show()
    
    plt.title(f"Threshold vs. Grade {i} Specificity")
    plt.plot(thresholds, specificity_grade[i], "r", label = f"{specificity_grade[i][thresholds.index(0.5)]} at t=0.5")
    plt.legend(loc = "lower right")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel(f"Grade {i} Specificity")
    plt.xlabel("Threshold")
    plt.show()
    
    plt.title(f"Grade {i} Sensitivity vs. Specificity")
    plt.plot(sensitivity_grade[i], specificity_grade[i], "r", label=f"{(sensitivity_grade[i][thresholds.index(0.5)], specificity_grade[i][thresholds.index(0.5)])} at t=0.5")
    plt.legend(loc = "lower right")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel(f"Grade {i} Specificity")
    plt.xlabel(f"Grade {i} Sensitivity")
    plt.show()

if __name__ == "__main__":
  load_devices()
  models_cv(frame_level_exists=2)
  evaluate_models()
