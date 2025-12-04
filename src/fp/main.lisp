(load "preprocess.lisp")
(load "metrics.lisp")
(load "knn.lisp")
(load "logistic_regression.lisp")
(load "linear_regression.lisp")
(load "naive_bayes.lisp")
(load "id3.lisp")


(defun get-cli-arg (args flag default-val)
  "Finds a flag (e.g. '--k') in the args list and returns the next value.
  Returns default-val if the flag is missing."
  (let ((pos (position flag args :test #'equal)))
    (if (and pos (< (1+ pos) (length args)))
      (let ((val-str (nth (1+ pos) args)))
        ;; Try to parse as number, otherwise return string
        (let ((parsed (read-from-string val-str)))
          (if (numberp parsed) parsed val-str)))
      default-val)))


(defun process-target-column (row)
  (let* ((raw-val (nth 13 row))
         (val-str (format nil "~a" raw-val))
         (clean-val (string-trim '(#\Space #\Tab #\Newline #\Return) val-str)))
    (if (string= clean-val "<=50K") 0 1)))

;; CHANGED: Now accepts a filename argument
(defun load-and-prep-data (filename)
  (format t "Loading and preparing data from: ~a... (This takes a moment)~%" filename)

  (let* (;; --- load csv ---
         (all-string-data (load-csv filename))
         (data-size (length all-string-data))
         (all-mixed-data (mapcar #'convert-row-to-mixed-types all-string-data))
         (ohe-maps (build-ohe-maps all-mixed-data))

         ;; ============================================================
         ;; For KNN, Logistic Regression, Naive Bayes.
         ;; - Removes 'Income' (Col 13).
         ;; - Keeps 'Hours' (Col 11).
         ;; - Uses One-Hot Encoding.
         ;; ============================================================
         (all-X-classification
           (mapcar #'(lambda (row) (apply-ohe (subseq row 0 13) ohe-maps)) all-mixed-data))

         ;; ============================================================
         ;; For Linear Regression
         ;; - Removes 'Income' (Col 13).
         ;; - Uses One-Hot Encoding.
         ;; ============================================================
         (all-X-regression
           (mapcar #'(lambda (row)
                       (let ((masked-row (subseq row 0 13)))
                         (setf (nth 11 masked-row) 0) (apply-ohe masked-row ohe-maps))) all-mixed-data))

         ;; ============================================================
         ;; For ID3 Algorithm.
         ;; - Removes 'Income' (Col 13).
         ;; - Uses RAW MIXED TYPES (Strings & Numbers), no One-Hot.
         ;; ============================================================
         (all-X-id3 (mapcar #'(lambda (row) (subseq row 0 13)) all-mixed-data))

         ;; --- CREATE TARGETS (Y) ---
         (all-y-classification (mapcar #'process-target-column all-mixed-data))
         (all-y-regression (mapcar #'(lambda (r) (float (nth 11 r))) all-mixed-data))

         ;; --- SPLIT INDICES (80/20) ---
         (index-list (create-index-list data-size))
         (rng (sb-ext:seed-random-state 42))
         (shuffled-indices (shuffle-list index-list rng))
         (split-point (floor (* 0.8 data-size)))
         (train-indices (subseq shuffled-indices 0 split-point))
         (test-indices (subseq shuffled-indices split-point))

         ;; --- BUILD SUBSETS ---

         ;; Classification Subsets (Train/Test)
         (x-classification-train (build-subset all-X-classification train-indices))
         (x-classification-test  (build-subset all-X-classification test-indices))

         ;; Regression Subsets (Train/Test)
         (x-regression-train (build-subset all-X-regression train-indices))
         (x-regression-test  (build-subset all-X-regression test-indices))

         ;; ID3 Subsets (Train/Test)
         (x-id3-train (build-subset all-X-id3 train-indices))
         (x-id3-test  (build-subset all-X-id3 test-indices))

         ;; Target Subsets
         (y-classification-train (build-subset all-y-classification train-indices))
         (y-classification-test  (build-subset all-y-classification test-indices))
         (y-regression-train (build-subset all-y-regression train-indices))
         (y-regression-test  (build-subset all-y-regression test-indices))

         ;; --- NORMALIZATION (Z-Score) ---
         (stats-class (calculate-column-stats x-classification-train))
         (norm-x-class-train (mapcar #'(lambda (r) (apply-normalization-to-row r stats-class)) x-classification-train))
         (norm-x-class-test (mapcar #'(lambda (r) (apply-normalization-to-row r stats-class)) x-classification-test))

         (stats-reg (calculate-column-stats x-regression-train))
         (norm-x-reg-train (mapcar #'(lambda (r) (apply-normalization-to-row r stats-reg)) x-regression-train))
         (norm-x-reg-test (mapcar #'(lambda (r) (apply-normalization-to-row r stats-reg)) x-regression-test)))

    ;; Return Package
    (list :x-class-train norm-x-class-train :x-class-test norm-x-class-test
          :x-reg-train norm-x-reg-train   :x-reg-test norm-x-reg-test
          :x-id3-train x-id3-train        :x-id3-test x-id3-test
          :y-class-train y-classification-train :y-class-test y-classification-test
          :y-reg-train y-regression-train       :y-reg-test y-regression-test)))

(defun run-knn (data k)
    (format t "Algorithm: KNN~%")

  (let* (
         (x-train (getf data :x-class-train))
         (y-train (getf data :y-class-train))

         ;; Renaming original full lists so we can slice them
         (x-full  (getf data :x-class-test))
         (y-full  (getf data :y-class-test))

         (x-test (subseq x-full 0 (min (length x-full) 100)))
         (y-test (subseq y-full 0 (min (length y-full) 100)))

         ;; --- DEFINE VARIABLES (Init to 0/nil) ---
         (start 0)
         (end 0)
         (time 0.0)
         (sloc 0)
         (preds nil))

    ;; start timer
    (setf start (get-internal-real-time))

    ;; run algo
    (setf preds (mapcar #'(lambda (row) (knn x-train y-train row k)) x-test))

    ;; end timer
    (setf end (get-internal-real-time))

    ;; calc and print
    (setf time (/ (- end start) internal-time-units-per-second))

    (format t "Train time: ~,4f seconds~%" time)
    (format t "KNN Accuracy: ~5f~%" (calc-accuracy preds y-test))
    (format t "Metric 3 SLOC: ~a~%" (get-sloc "knn.lisp"))


    ))

(defun run-logistic-regression (data learning-rate epochs l2-penalty)
  (format t "Algorithm: Logistic Regression~%")
  (let* ((x-train (getf data :x-class-train))
         (y-train (getf data :y-class-train))
         (x-test  (getf data :x-class-test))
         (y-test  (getf data :y-class-test))

         ;; 1. DEFINE VARIABLES (Init to 0/nil)
         (start 0) (end 0) (time 0.0)
         (model nil)
         (w nil)
         (b nil)
         (preds nil)
         (sloc 0)
         )

    ;; start timer
    (setf start (get-internal-real-time))

    ;; run model
    (setf model (train-logistic x-train y-train learning-rate epochs l2-penalty))

    ;; stop timer
    (setf end (get-internal-real-time))
    (setf time (/ (- end start) internal-time-units-per-second))

    ;; EXTRACT WEIGHTS & PREDICT (Using FULL x-test)
    (setf w (first model))
    (setf b (second model))
    (setf preds (mapcar #'(lambda (row) (logistic-predict row w b)) x-test))

    ;; print
    (format t "Train time: ~,4f seconds~%" time)
    (format t "Metric 1 Accuracy: ~5f~%" (calc-accuracy preds y-test))
    (format t "Metric 2 Macro-F1: ~5f~%" (calc-macro-f1 preds y-test))
    ;;(format t "Metric 3 SLOC: ~a~%" sloc)


    (format nil "Acc: ~,3f" (calc-accuracy preds y-test))
    (format nil "F1: ~,3f" (calc-macro-f1 preds y-test))
    (format t "Metric 3 SLOC: ~a~%" (get-sloc "logistic_regression.lisp"))

    ))

(defun run-linear-regression (data l2-penalty)
  (format t "Algorithm: Linear Regression ~%")
  (let* ((x-train (getf data :x-reg-train))
         (y-train (getf data :y-reg-train))
         (x-test  (getf data :x-reg-test))
         (y-test  (getf data :y-reg-test))

         (start 0)
         (end 0)
          (time 0.0)
         (weights nil)
         (preds nil)
          (sloc 0)
)


    ;; start timer
    (setf start (get-internal-real-time))

    ;; run algo
    (setf weights (train-linear x-train y-train l2-penalty))

    ;; stop timer
    (setf end (get-internal-real-time))
    (setf time (/ (- end start) internal-time-units-per-second))

    ;; run prediction
    (setf preds (mapcar #'(lambda (row) (predict-linear row weights)) x-test))

    ;; print results
    (format t "Train time: ~,4f seconds~%" time)
    (format t "Metric 1 RMSE: ~5f~%" (calc-rmse preds y-test))
    (format t "Metric 2 R^2:  ~5f~%" (calc-r2 preds y-test))
    (format t "Metric 3 SLOC: ~a~%" (get-sloc "linear_regression.lisp"))


))

(defun run-id3 (data max-depth n-bins)
  (format t "Algorithm: Decision Tree~%")
  (let* ((x-train (getf data :x-id3-train))
         (y-train (getf data :y-class-train))
         (x-test  (getf data :x-id3-test))
         (y-test  (getf data :y-class-test))


         (start 0)
         (end 0)
         (time 0.0)
         (tree nil)
         (preds nil)
         (sloc 0)
         (cols (loop :for i :below 13 :collect i)))

    ;; start timer
    (setf start (get-internal-real-time))

    ;; run algo
    (setf tree (id3-train x-train y-train cols 0 max-depth n-bins))

    ;; stop timer
    (setf end (get-internal-real-time))
    (setf time (/ (- end start) internal-time-units-per-second))

    ;; run prediction
    (setf preds (mapcar #'(lambda (row) (predict-id3-row row tree n-bins)) x-test))

    ;; print
    (format t "Train time: ~,4f seconds~%" time)
    (format t "Metric 1 Accuracy: ~5f~%" (calc-accuracy preds y-test))
    (format t "Metric 2 Macro-F1: ~5f~%" (calc-macro-f1 preds y-test))
    (format t "Metric 3 SLOC: ~a~%" (get-sloc "id3.lisp"))
    ))

(defun run-naive (data)
  (format t "Algorithm: Naive Bayes~%")

  (let* ((x-train (getf data :x-class-train))
         (y-train (getf data :y-class-train))
         (x-test  (getf data :x-class-test))
         (y-test  (getf data :y-class-test))

         (start 0)
         (end 0)
         (time 0.0)
         (model nil)
          (preds nil)
          (sloc 0)
          )


    ;; start timer
    (setf start (get-internal-real-time))

    ;; run algo
    (setf model (train-nb x-train y-train))

    ;; stop
    (setf end (get-internal-real-time))
    (setf time (/ (- end start) internal-time-units-per-second))

    ;; run prediction
    (setf preds (mapcar #'(lambda (row) (predict-nb-row row model)) x-test))

    ;; print
    (format t "Train time: ~,4f seconds~%" time)
    (format t "Metric 1 Accuracy: ~5f~%" (calc-accuracy preds y-test))
    (format t "Metric 2 Macro-F1: ~5f~%" (calc-macro-f1 preds y-test))
    (format t "Metric 3 SLOC: ~a~%" (get-sloc "naive_bayes.lisp"))

))



(defun handle-cli-args ()
  (let ((args sb-ext:*posix-argv*))

    ;; Check if running in Script Mode (Arguments present)
    (if (> (length args) 1)
        (progn
          ;; CHANGED: Read --train flag, default to standard path
          (let* ((train-file (get-cli-arg args "--train" "../data/adult_income_cleaned.csv"))
                 (data (load-and-prep-data train-file))
                 ;; 2. Extract the Algorithm Name
                 (algo (get-cli-arg args "--algo" "")))

            (cond
              ;; --- KNN ---
              ;; Command: --algo knn --k (int)
              ((string-equal algo "knn")
               (let ((k (get-cli-arg args "--k" 2)))
                 (run-knn data k)))

              ;; --- LOGISTIC ---
              ;; Command: --algo logistic --lr (float)--epochs (int) --l2(float)
              ((string-equal algo "logistic")
               (let ((lr     (get-cli-arg args "--lr" 0.2))     ;; .2 default
                     (epochs (get-cli-arg args "--epochs" 200)) ;; 200 default
                     (l2     (get-cli-arg args "--l2" 0.03)))  ;; .03 default
                 (run-logistic-regression data lr epochs l2)))

              ;; --- LINEAR ---
              ;; Command: --algo linear --l2 (float)
              ((string-equal algo "linear")
               (let ((l2 (get-cli-arg args "--l2" 0.2))) ;; .2 default
                 (run-linear-regression data l2)))

              ;; --- DECISION TREE ---
              ;; Command: --algo tree --max_depth (int) --n_bins (int)
              ((string-equal algo "tree")
               (let ((depth (get-cli-arg args "--max_depth" 5)) ;; 5 default
                     (bins  (get-cli-arg args "--n_bins" 10)))  ;; 10 default
                 (run-id3 data depth bins)))

              ;; --- NAIVE BAYES ---
              ;; Command: --algo nb
              ((string-equal algo "nb")
               (run-naive data)))))

      (main-menu))))

(handle-cli-args)
