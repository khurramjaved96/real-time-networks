import FlexibleNN

model_CAN = FlexibleNN.ContinuallyAdaptingNetwork(0.000001, 84*84, 1, 0)
model_CAN.set_input_values(np.eye(84).flatten())
model_CAN.step()
target = 1 + 0.98 * model_CAN.read_output_values()[0]
error = model_CAN.introduce_targets([target], 0.98, 0.99, [0])
for _ in range(20):
    model_CAN.add_feature(0.000001)
