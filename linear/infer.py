import onnx
import onnxruntime
import torch


if __name__ == "__main__":
    # verify onnx model
    onnx_file_name = "linear2d_onnx_model.onnx"
    onnx_model = onnx.load(onnx_file_name)

    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s"%e)
        exit(1)
    else:
        print("The model is valid!")

    infer_session = onnxruntime.InferenceSession(onnx_file_name)
    input_name = infer_session.get_inputs()[0].name
    output_name = infer_session.get_outputs()[0].name

    print("input name: ", input_name)
    print("output name: ", output_name)

    X = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])
    X = X.half()
    print(X.shape)

    result = infer_session.run([output_name], {input_name: X.numpy()})

    print("result: ", result)


