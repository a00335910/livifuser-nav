# Independent simulation-supervisor decision and terminal state.
std_msgs/Header header
string variant
uint32 seed
uint64 context_sequence
uint64 proposal_receipt_monotonic_ns
float32 raw_linear_x
float32 raw_angular_z
float32 clipped_linear_x
float32 clipped_angular_z
float32 executed_linear_x
float32 executed_angular_z
bool proposal_valid
bool inference_ready
bool stale_timeout
bool collision
bool uncertainty_intervention
bool success
bool terminal
string priority_reason
string terminal_reason
float64 ground_truth_clearance_m
float64 ground_truth_goal_distance_m
uint32 consecutive_success_samples
float64 proposal_age_ms
float64 control_interval_ms
# Ticks recovered from a stretched control interval, per amendment section 3.
# An episode that limps must be visible in the evidence, not smoothed over.
uint32 stretched_interval_count
