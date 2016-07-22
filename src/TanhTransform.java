import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.neural.NeuralUtils;


public class TanhTransform implements TransformFunction {

	@Override
	public SimpleMatrix value(SimpleMatrix input) {
		// TODO Auto-generated method stub
		return NeuralUtils.elementwiseApplyTanh(input);
	}

	@Override
	public SimpleMatrix grad(SimpleMatrix input) {
		// TODO Auto-generated method stub
		return NeuralUtils.elementwiseApplyTanhDerivative( input);
	}

	@Override
	public String getName() {
		// TODO Auto-generated method stub
		return "Tanh";
	}

}
