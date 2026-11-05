# NER classification report on the frozen held-out set
100 human-verified utterances. Exact (start, end, label) span matching. `micro avg` pools all spans (the span-level counterpart of accuracy); `macro avg` is the unweighted mean over labels with support. HEADING (3) and SQUAWK (1) have too few spans for their scores to mean much.

## LLM teacher (Qwen3.5-9B production labels)

```
              precision    recall  f1-score   support

    CALLSIGN       0.93      0.89      0.91        91
     COMMAND       0.93      0.91      0.92        86
    FACILITY       0.78      0.85      0.82        34
    ALTITUDE       1.00      0.88      0.94        26
    WAYPOINT       0.72      0.75      0.73        24
   FREQUENCY       1.00      0.61      0.76        23
      RUNWAY       1.00      1.00      1.00         7
     HEADING       1.00      0.33      0.50         3
      SQUAWK       1.00      1.00      1.00         1

   micro avg       0.90      0.85      0.88       295
   macro avg       0.93      0.80      0.84       295
weighted avg       0.91      0.85      0.88       295
```

## Student: spacy-balanced

```
              precision    recall  f1-score   support

    CALLSIGN       0.87      0.89      0.88        91
     COMMAND       0.95      0.84      0.89        86
    FACILITY       0.76      0.85      0.81        34
    ALTITUDE       1.00      0.85      0.92        26
    WAYPOINT       0.78      0.75      0.77        24
   FREQUENCY       0.69      0.96      0.80        23
      RUNWAY       1.00      1.00      1.00         7
     HEADING       0.67      0.67      0.67         3
      SQUAWK       1.00      1.00      1.00         1

   micro avg       0.86      0.86      0.86       295
   macro avg       0.86      0.87      0.86       295
weighted avg       0.87      0.86      0.86       295
```
