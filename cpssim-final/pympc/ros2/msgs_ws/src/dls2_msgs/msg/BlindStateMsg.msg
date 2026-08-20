# Header
string frame_id
uint32 sequence_id
float64 timestamp

string robot_name

string[12] joints_name
float64[12] joints_position
float64[12] joints_velocity
float64[12] joints_acceleration
float64[12] joints_effort
float64[12] joints_temperature

float64[4] feet_contact
float64[12] current_feet_positions

float64[] feet_positions            # Feet positions in world frame (12: 3D × 4 feet)
float64[] feet_velocities           # Feet velocities in world frame (12: 3D × 4 feet)
float64[] hip_positions             # Hip positions in world frame (12: 3D × 4 hips)
float64[] mass_matrix               # Mass matrix (144: 12x12)
float64[] bias_forces               # Bias forces (12: coriolis + gravity forces)
float64[] feet_jacobians            # Jacobians for 4 feet (144: 4 × 3x12)
float64[] feet_jacobian_dots        # Jacobian derivatives for 4 feet (144: 4 × 3x12, currently zero)
float64[] base_inertia
