# Robot-local acquisition lifecycle. Published on /livifuser/episode_state and
# recorded with every protocol-clean episode.
std_msgs/Header header

string episode_id
string phase
string reason
bool motion_permitted
float32 recording_elapsed_s
float32 recording_remaining_s
uint32 sequence
