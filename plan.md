создать отдельную модель под каждую пользотельскую конфигурацию моделей (выбор набора данных под классификацию)
изменение дизайна под черно белый "ascii арт" с анимациями
парсер реддита должен проверить оценку заведомо sfw контента перед добавление в датасет
CI/CD

Мск должно быть в итоге просто шлюзом


    # Корневая директория сайта
    root /var/www/blackfox;
    index index.html;

18 11 2025
Generating final predictions for reports & SHAP data:  73%|███████████████████████████████████████████████████████████████████████████████████████████████████▋                                    | 187/255 [02:31<01:48,  1.59s/it]Premature end of JPEG file
Generating final predictions for reports & SHAP data:  74%|████████████████████████████████████████████████████████████████████████████████████████████████████▊                                   | 189/255 [02:32<01:06,  1.00s/it]libpng warning: sBIT: invalid
Generating final predictions for reports & SHAP data:  84%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                     | 215/255 [02:51<00:20,  1.99it/s]libpng warning: iCCP: known incorrect sRGB profile
Generating final predictions for reports & SHAP data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 255/255 [03:16<00:00,  1.30it/s]
Saved final test data paths to model/resnet/results/final_test_data_paths.txt
Saved final test data labels to model/resnet/results/final_test_data_labels.txt
Optimal threshold based on F1-score on test set: 0.8489 (F1: 0.9597)
Saved optimal threshold (0.8489) to model/resnet/optimal_threshold.pkl

--- Metrics Report for final_test_metrics_report_optimal_thresh.txt ---
              precision    recall  f1-score   support

     Regular       0.99      0.99      0.99    102794
        NSFW       0.96      0.96      0.96     27591

    accuracy                           0.98    130385
   macro avg       0.97      0.97      0.97    130385
weighted avg       0.98      0.98      0.98    130385

ROC-AUC Score: 0.9972
Average Precision Score: 0.9921
Confusion matrix saved to model/resnet/results/final_test_confusion_matrix_optimal_thresh.png
ROC curve saved to model/resnet/results/final_test_roc_curve.png
Precision-Recall curve saved to model/resnet/results/final_test_precision_recall_curve.png

--- Exporting and Quantizing Model ---
Using best saved model from model/resnet/best_resnet_state_dict.pth for export and quantization.
Exporting model to ONNX: model/resnet/nsfw_resnet.onnx
Error during ONNX export or check: No module named 'onnxscript'

Testing ONNX inference...
ONNX inference test - Logits shape: (1, 1), Probs (example): [0.08826482]

--- Script Finished ---
    ~/projects/blackfox    main !3 ?3 ▓▒░
