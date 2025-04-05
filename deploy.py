# Save model (for Raspberry Pi/drone)
torch.save(model.state_dict(), "fire_detector.pth")

# Convert to ONNX (faster inference)
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "fire_detector.onnx")
