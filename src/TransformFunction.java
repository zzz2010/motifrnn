import org.ejml.simple.SimpleMatrix;


public interface TransformFunction {

	public SimpleMatrix value(SimpleMatrix input);
	
	public SimpleMatrix grad(SimpleMatrix input);
	
	public String getName();
	
}
