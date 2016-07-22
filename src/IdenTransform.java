import org.ejml.simple.SimpleMatrix;


public class IdenTransform implements TransformFunction {

	@Override
	public SimpleMatrix value(SimpleMatrix input) {
		// TODO Auto-generated method stub
		return Utils.elementwiseApplyIden(input);
	}

	@Override
	public SimpleMatrix grad(SimpleMatrix input) {
		// TODO Auto-generated method stub
		return Utils.elementwiseApplyIdenDerivative(input);
	}

	@Override
	public String getName() {
		// TODO Auto-generated method stub
		return "Iden";
	}

}
