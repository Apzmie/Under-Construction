# Under-Construction

## Critic Loss
The initial numbers before calculating the error are just set randomly for the example.
![smooth_l1_loss](images/smooth_l1_loss.png)
1. Sort `next_quantiles` in ascending order, truncate the top few, and input it into the TD equation to get `target_quantiles`, with `pred_quantiles` ready as well.
2. Because of truncation, `target_quantiles` and `pred_quantiles` have different numbers of elements, but subtract by automatically matching their shapes, then each element of one array meets each element of the other.
3. Although averaging all the error elements works as a loss, doing so ends up matching the median of `target_quantiles`, losing the meaning of quantiles. So for the next step, smooth L1 loss is applied, smoothing out the sharp L1 graph to make it differentiable, which results in smaller values overall. This is applied so that when the error is large, it avoids being affected by extreme values without squaring the error like L2 loss, and when the error is small, it makes the values even smaller to weaken the critic's update intensity so it can learn the quantiles more precisely.

![critic_loss](images/critic_loss.png)
4.
