# Results: synthetic_iq_jamming

| method | task | test_acc | val_acc | ood_acc | res_rate | out_rate | trainable_syn | train_s |
|---|---|---|---|---|---|---|---|---|
| gru | classification | 1 | 1 | 0.9425 |  |  |  | 1.901 |
| linear | classification | 0.9975 | 0.9967 | 0.7325 | 0.2197 |  |  | 2.499 |
| rstdp | classification | 0.8 | 0.81 | 0.625 | 0.233 | 0.1194 | 400 | 16.87 |
| surrogate | classification | 0.9975 | 0.9967 | 0.7725 | 0.233 | 0.2019 | 400 | 5.413 |
