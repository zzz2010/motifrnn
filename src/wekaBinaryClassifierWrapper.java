import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.TreeMap;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import ml.options.Options;
import ml.regression.LASSO;
import ml.regression.Regression;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RemoveUseless;
import edu.uci.lasso.LassoFit;
import edu.uci.lasso.LassoFitGenerator;


public class wekaBinaryClassifierWrapper extends Classifier {

	/**
	 * 
	 */
	private static final long serialVersionUID = 84732258783789128L;

	Classifier baseLearner;
	HashMap<String,Float> FeatureImportance;
	int maxNumofFeature;
	FilteredClassifier finalLearner;
	ChiSquaredAttributeEval weka_eval=null;
	
	public wekaBinaryClassifierWrapper(Classifier baseLearner,int maxNumofFeature) {
		super();
		this.maxNumofFeature=maxNumofFeature;
		this.baseLearner = baseLearner;
		FeatureImportance=new HashMap<String, Float>();
		finalLearner=null;
	}
	


	public  Filter LassoSelection2(Instances arg0,int maxNumofFeature)
	{
		FeatureImportance.clear();
		Options options = new Options();
		options.maxIter = 600;
		options.nTopTerm=maxNumofFeature;
		options.lambda = 0.05;
		options.verbose = !true;
		options.calc_OV = !true;
		options.epsilon = 1e-5;
		 int featuresCount=arg0.numAttributes()-1;
		 int numObservations = arg0.numInstances();
		 double[][] data=new double[numObservations][featuresCount];
		Matrix depVars=new DenseMatrix(numObservations, 1);
		
		 for (int j = 0; j < arg0.numInstances(); j++) {
			 double[] inst=arg0.instance(j).toDoubleArray();
			 double target= arg0.instance(j).classValue();
			 for (int k = 0; k < inst.length-1; k++) {
				 data[j][k]=inst[k];
			 }
			 depVars.setEntry(j, 0, target);
		 }
		Regression LASSO = new LASSO(options);
		LASSO.feedData(data);
		LASSO.feedDependentVariables(depVars);
		
		LASSO.train();
		TreeMap<Double,String> sorted_features=new TreeMap<Double,String>();
		Random rand=new Random(1);
		 for (int i = 0; i < arg0.numAttributes()-1; i++) {
			 double b = LASSO.W.getEntry(i, 0);
			 String featName=arg0.attribute(i).name();
			 if(b==0)
				 continue;
			 if(featName=="target_value")
				 continue;
			 sorted_features.put(Math.abs(b)+rand.nextDouble()/10000, (i+1)+"\t"+featName+"\t"+b);
		 }
		 String filterStr="";
		 for (Double score : sorted_features.descendingKeySet()) {
			 String[] fstr=sorted_features.get(score).split("\t");
			 FeatureImportance.put(fstr[1], Float.valueOf(fstr[2]));
			 filterStr+=","+fstr[0];
			 if(FeatureImportance.size()==maxNumofFeature)
				 break;
		}
		 filterStr+=","+(featuresCount+1);
		 weka.filters.unsupervised.attribute.Remove filter=new Remove();
			filter.setAttributeIndices(filterStr);
			filter.setInvertSelection(true);
		 
			try {
				filter.setInputFormat(arg0);
				return filter;
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		return filter;
		
	}
	public  Filter LassoSelection(Instances arg0,int maxNumofFeature)
	{
		FeatureImportance.clear();
		 LassoFitGenerator fitGenerator = new LassoFitGenerator();
		 int featuresCount=arg0.numAttributes()-1;
		 int numObservations = arg0.numInstances();
		 try {
			fitGenerator.init(featuresCount, numObservations);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		System.out.println(arg0.attribute(arg0.numAttributes()-1).name());
		 for (int j = 0; j < arg0.numInstances(); j++) {
			 double[] inst=arg0.instance(j).toDoubleArray();
			 float target=(float) arg0.instance(j).classValue();
			 float[] observation=new float[inst.length-1];
			 for (int k = 0; k < observation.length; k++) {
				 observation[k]=(float) inst[k];
			}
			  fitGenerator.setObservationValues(j, observation);
             fitGenerator.setTarget(j, target);
		}
		 
		  LassoFit fit = fitGenerator.fit(maxNumofFeature);
		  String filterStr="";
		 
		 for (int i = 0; i <  fit.nonZeroWeights[fit.numberOfLambdas-1]; i++) {
				if(filterStr.length()>0)
					filterStr+=",";
				filterStr+=fit.indices[i]+1;
				FeatureImportance.put(arg0.attribute(fit.indices[i]).name(), (float) fit.getWeights(fit.numberOfLambdas-1)[fit.indices[i]]);
		}
		 filterStr+=","+(featuresCount+1);
		 weka.filters.unsupervised.attribute.Remove filter=new Remove();
			filter.setAttributeIndices(filterStr);
			filter.setInvertSelection(true);
		 
			try {
				filter.setInputFormat(arg0);
				return filter;
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		return filter;
	}

	public  Filter WekaSelection(Instances arg0,int maxNumofFeature)
	{
	
		 
		weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
		
		
//		CfsSubsetEval eval=new CfsSubsetEval();
//		GreedyStepwise search =new GreedyStepwise();
//		search.setSearchBackwards(true);
		
		
		weka_eval=new ChiSquaredAttributeEval();
		Ranker search=new Ranker();
		search.setNumToSelect(maxNumofFeature);
		
		
		  filter.setEvaluator(weka_eval);
		    filter.setSearch(search);
		    
		    try {
				filter.setInputFormat(arg0);
				return filter;
		    }
		    catch(Exception e1) 
		    {
		    	e1.printStackTrace();
		    }
		    
//		    for (int i = 0; i <data2.numAttributes(); i++) {
//				System.err.println(data2.attribute(i).name());
//			}
		  return filter;
	}
	@Override
	public double classifyInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub

		return finalLearner.classifyInstance(arg0);
	}



	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		return finalLearner.distributionForInstance(arg0);
	}



	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		 Capabilities result = super.getCapabilities(); 
		 result.enable(Capability.NOMINAL_CLASS);
		 result.enable(Capability.NOMINAL_ATTRIBUTES);
		    result.enable(Capability.NUMERIC_ATTRIBUTES);
		return result;
	}





	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		// TODO Auto-generated method stub
		weka.filters.unsupervised.attribute.RemoveUseless featSel=new RemoveUseless();
//		weka.filters.unsupervised.attribute.PrincipalComponents featSel=new PrincipalComponents();
		
//		Filter featSel=WekaSelection(arg0, maxNumofFeature);//LassoSelection2   WekaSelection
		finalLearner=new FilteredClassifier();
		finalLearner.setClassifier(baseLearner);
		finalLearner.setFilter(featSel);
		finalLearner.buildClassifier(arg0);
//		System.err.println(baseLearner.toString());
		
		if(baseLearner.getClass()==LibLINEAR2.class)
		{
			Map<String, Float> sortedMap = Utils.sortByAbsValue(((LibLINEAR2)baseLearner).getFeatureWeights());
			Iterator<Entry<String, Float>> iter = sortedMap.entrySet().iterator();
			while(iter.hasNext())
			{
				Entry<String, Float> tmp = iter.next();
				String featureName=tmp.getKey()	;	
				if(Utils.displayOnlyMotif&&( featureName.contains("Repeat")||Utils.checkCoFounder(featureName)))
					continue;	
				FeatureImportance.put(tmp.getKey()	, tmp.getValue());
				 if(FeatureImportance.size()==maxNumofFeature)
					 break;
			}
		}
		
		//if not lasso
		if(FeatureImportance.size()==0)
		{
			TreeMap<Double,Integer> sorted_features=new TreeMap<Double,Integer>();
//			PearsonsCorrelation pcorr=new PearsonsCorrelation();
			weka_eval=new ChiSquaredAttributeEval();
			weka_eval.buildEvaluator(arg0);
			double[] targetValues=arg0.attributeToDoubleArray( arg0.numAttributes()-1);
			 for (int i = 0; i < arg0.numAttributes()-1; i++) {
				 double b = weka_eval.evaluateAttribute(i);
				 if(b>0)
				 {
//				 String featName=arg0.attribute(i).name();
				 sorted_features.put(b, i);
				 }
			 }
			 for (Double score : sorted_features.descendingKeySet()) {
				 int featid=sorted_features.get(score);
				 double[] featVals=arg0.attributeToDoubleArray(featid);
				 double sign=0;
				 for (int i = 0; i < featVals.length; i++) {
					if(featVals[i]*targetValues[i]>0)
						sign+=featVals[i];
					else
						sign-=featVals[i];
				}
				 if(sign>0)
					 FeatureImportance.put(arg0.attribute(featid).name(), score.floatValue());	
				 else
					 FeatureImportance.put(arg0.attribute(featid).name(), -score.floatValue());	
				 if(FeatureImportance.size()==maxNumofFeature)
					 break;
			}
		}
		
	}

}
