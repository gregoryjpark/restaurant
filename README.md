#restaurant
### My best submission for the TFI Restaurant Revenue Kaggle Competition

`tfi_submission.R` is an R script that recreates my best single set of predictions for this competition.  

#### Preprocessing
* `Open.Date` was transformed into several features representing date, month, day of month, and year.
* `City.Group` and `Type` were transformed into several binary features, one for each observed value.
* Columns `P1` through `P37` were also transformed into binary features, because it seemed possible that they were actually categorical features, not continuous as I originally thought. In any case, I kept the original variables as well.
* `revenue`, the response, was log-transformed during training, but I evaluated the predictions after back-transforming to the original scale.

#### Model
After creating additional features, I trained a random forest with 5,000 trees. My thinking was that most of the new features were irrelevant, but a random forest wouldn't be as sensitive to irrelevant features if I used many, many trees. I used `caret` to choose the `mtry` parameter (`mtry` = 225) through cross-validation.

#### Result
This single model would have placed around ~150th place (out of ~2,200). My final ranking was a bit lower (202nd) because I combined this model with another (a random forest with a much smaller feature set) and that performed slightly worse.

