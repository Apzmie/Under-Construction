# Under-Construction

## Critic Loss
The initial numbers before calculating the error are just set randomly for the example.
![smooth_l1_loss](images/smooth_l1_loss.png)

1. Sort `next_quantiles` in ascending order, truncate the top few, and input it into the TD equation to get `target_quantiles`, with `pred_quantiles` ready as well.
2. Because of truncation, `target_quantiles` and `pred_quantiles` have different numbers of elements, but subtract by automatically matching their shapes, then each element of one array meets each element of the other.
3. 
