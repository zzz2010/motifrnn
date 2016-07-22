import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import libsvm.svm;
import ml.options.Options;
import ml.regression.LASSO;
import ml.regression.Regression;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.RBFNetwork;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import de.bwaldvogel.liblinear.SolverType;
import edu.uci.lasso.LassoFit;
import edu.uci.lasso.LassoFitGenerator;


public class wekaLearningThread implements Runnable {

	wekaBinaryClassifierWrapper learner;
	Instances training_data;
	int class_id;
	List<String> selectedFeature;
	
	public wekaLearningThread(Instances samples,int class_id)
	{
//		LibSVM svm=new LibSVM();
//		svm.setShrinking(true);
//		svm.setNormalize(true);
//		svm.setProbabilityEstimates(true);
//		svm.setDebug(false);
//		
////		svm.setCacheSize(1000);
//		svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
//		svm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_C_SVC,LibSVM.TAGS_SVMTYPE));
		
		
		LibLINEAR2 baseClassifier=new  LibLINEAR2();
		baseClassifier.setProbabilityEstimates(true);
		baseClassifier.setCost(Utils.regCoef);
//		baseClassifier.setNormalize(true);
		baseClassifier.setSVMType(new SelectedTag(SolverType.L2R_LR.getId(),LibLINEAR2.TAGS_SVMTYPE));
		System.err.println(baseClassifier.getSolverType());
		
		learner=new wekaBinaryClassifierWrapper( baseClassifier,Utils.numTopFeatures); //40 motifs selected
//		learner=new wekaBinaryClassifierWrapper( new AdaBoostM1(),Utils.numTopFeatures); 
		
//		AdaBoostM1 adaboost=new AdaBoostM1();
//		adaboost.setClassifier(baseClassifier);
////		adaboost.setNumIterations(3);
//		System.err.println("Use AdaBoost");
//		learner=new wekaBinaryClassifierWrapper(adaboost,Utils.numTopFeatures); 
		
		this.training_data=samples;
		this.class_id=class_id;
		System.err.println("baselearner type: "+learner.baseLearner.toString());
	}
	
	
	
	@Override
	public void run() {
		
		if(Utils.studyGC_CpG_Repeat) //output the prediction of CG, CpG, Repeat at the same time
		{
			run_CG_CpG_Repeat() ;
			return ;
		}
		// TODO Auto-generated method stub
		try {
			//to do , lasso regression to reduce dimension
			selectedFeature=new ArrayList<String>();
			System.err.println("building simple model for class "+class_id);
			
			//trick to speed up the debugging
//			System.err.println("PCAing........");
//			weka.filters.unsupervised.attribute.PrincipalComponents featSel=new weka.filters.unsupervised.attribute.PrincipalComponents();
//			featSel.setCenterData(true);
//			featSel.setMaximumAttributes(100);
//			featSel.setInputFormat(training_data);
//			training_data=Filter.useFilter(training_data, featSel);
//			System.err.println("PCA finish");
			//need to remove it in real application
			
			Evaluation eval1 = new Evaluation(training_data);
			int nfold=5;
			if(training_data.numInstances()<1000)
				nfold=training_data.numInstances()/10; //leave two out
		
			if(!Utils.noCV)
			{
				
			System.err.println(nfold+" folds cross validations:");
			eval1.crossValidateModel(learner, training_data, nfold, new Random(1));

			}
			
			learner.buildClassifier(training_data);
			if(Utils.noCV)
			{
				eval1.evaluateModel(learner, training_data);
			}
			System.err.println(eval1.toSummaryString(true));
			System.err.println(eval1.toMatrixString());
			double auc = eval1.areaUnderROC(1);
			
			String classname="class "+class_id;
			if(MotifSimpleModel.ClassNames!=null&&class_id<MotifSimpleModel.ClassNames.size())
				classname=MotifSimpleModel.ClassNames.get(class_id);
		  
		  Map<String, Float> sortedFeatureWeight = Utils.sortByValue(learner.FeatureImportance);
		  
		  for (Entry<String, Float> feat : sortedFeatureWeight.entrySet()) {
			  selectedFeature.add(feat.getKey()+":"+feat.getValue());
//			  double score=DataSelector.kmerPrior.get(feat.getKey());
//			  if(score<0)
//			  {
//				  System.out.println("unconserved: "+feat.getKey()+" "+score);
//			  }
		}
		 System.out.println(classname+" auc:"+auc+"\n"+selectedFeature);
		 
		 System.gc();
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	
	
	
	public void run_CG_CpG_Repeat() {
		Utils.displayOnlyMotif=false;
		Utils.excludeRepeat=false;
		String[] biasNames=new String[]{"Repeat","CpG","CG"};

		String printStr="";
		try {
			//to do , lasso regression to reduce dimension
			selectedFeature=new ArrayList<String>();
			System.err.println("building simple model for class "+class_id);
			
			
			Evaluation eval1 = new Evaluation(training_data);
			int nfold=5;
			if(training_data.numInstances()<1000)
				nfold=training_data.numInstances()/10; //leave two out
			eval1.crossValidateModel(learner, training_data, nfold, new Random(1));
			System.err.println(nfold+" folds cross validations:");
			System.err.println(eval1.toSummaryString(true));
			System.err.println(eval1.toMatrixString());
			double auc = eval1.areaUnderROC(1);
			String classname="class "+class_id;
			if(MotifSimpleModel.ClassNames!=null&&class_id<MotifSimpleModel.ClassNames.size())
				classname=MotifSimpleModel.ClassNames.get(class_id);
		  learner.buildClassifier(training_data);
		  Map<String, Float> sortedFeatureWeight = Utils.sortByValue(learner.FeatureImportance);
		  for (Entry<String, Float> feat : sortedFeatureWeight.entrySet()) {
			  selectedFeature.add(feat.getKey()+":"+feat.getValue());
		}
		  
		  printStr=classname+"\t"+"Repeat:"+auc;
//		 System.out.println(classname+" Repeat:"+auc+"\n"+selectedFeature);
		 
		 System.gc();
		 
		 for (int i = 0; i < biasNames.length-1; i++) {
			 String rmIdstr="";
				for (int j= 0; j < training_data.numAttributes(); j++) {
					if(training_data.attribute(j).name().contains(biasNames[i]))
					{
						if(rmIdstr!="")
							rmIdstr+=",";
						rmIdstr+=j+1;
					}
				}
				Remove filter=new Remove();
				filter.setAttributeIndices(rmIdstr);
				filter.setInputFormat(training_data);
				training_data=Filter.useFilter(training_data, filter);
				 eval1 = new Evaluation(training_data);
				
				 eval1.crossValidateModel(learner, training_data, 5, new Random(1));
				 auc = eval1.areaUnderROC(1);
				 printStr+="\t"+biasNames[i+1]+":"+auc;
		}
		 System.out.println(printStr+"\n"+selectedFeature);
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	public static void testLasso2()
	{
        Random rand=new Random(1);
		Options options = new Options();
		options.maxIter = 100;
		options.lambda = 0.05;
		options.verbose = !true;
		options.calc_OV = !true;
		options.epsilon = 1e-5;
	       int numObservations=1000;
	       int featuresCount=500;
		 double[][] data=new double[numObservations][featuresCount];
		Matrix depVars=new DenseMatrix(numObservations, 1);

	     for (int i = 0; i < numObservations; i++) {
	         float[] curObservation = new float[featuresCount];
	         for (int f = 0; f < featuresCount; f++) {
	        	   curObservation[f]=rand.nextFloat();
	        	   data[i][f] =curObservation[f];
	         }
	         depVars.setEntry(i, 0, curObservation[0]+curObservation[9]+curObservation[28]);
		}
			Regression LASSO = new LASSO(options);
			LASSO.feedData(data);
			LASSO.feedDependentVariables(depVars);

			LASSO.train();
	
			System.out.print(LASSO.W);

	}
	
	
	public static void testLiblinear()
	{
        Random rand=new Random(1);
        int featureNum=100;

		FastVector attrList=new FastVector(featureNum+1);
		for (int i = 0; i < featureNum; i++) {
			attrList.addElement(new Attribute("attr"+i));
		}


		FastVector Tvalues = new FastVector();
		Tvalues.addElement("negative");
		Tvalues.addElement("positive");
		Attribute label = new Attribute("target_value",Tvalues);
		attrList.addElement(label);
		
	       int numObservations=1000;
	       int featuresCount=featureNum+1;
	       
		     Instances weka_traindata=new Instances("testLiblinear", attrList,numObservations);
	     for (int i = 0; i < numObservations; i++) {
	         double[] curObservation = new double[featuresCount];
	         for (int f = 0; f < featuresCount-1; f++) {
	        	   curObservation[f]=rand.nextDouble();
	         }
	         double labelval=(4*curObservation[0]-curObservation[9]-curObservation[28]);
	         if(labelval>1)
	         {
	        	 labelval=1;
	         }
	         else
	         {
	        	 labelval=0;
	         }
	         curObservation[featureNum]=labelval;
	         Instance instance = new Instance(1, curObservation);
			 weka_traindata.add(instance);
		}
	     weka_traindata.setClassIndex(featureNum);
	     wekaLearningThread t=new wekaLearningThread(weka_traindata, 0);
			t.run();
	

	}
	
	public static void testLasso()
	{
        Random rand=new Random(1);
       List<float[]> observations = new ArrayList<float[]>();
       List<Float> targets = new ArrayList<Float>();
       int numObservations=1000;
       int featuresCount=500;
	     for (int i = 0; i < numObservations; i++) {
	         float[] curObservation = new float[featuresCount];
	         for (int f = 0; f < featuresCount; f++) {
	                 curObservation[f] =  rand.nextFloat();
	         }
	         observations.add(curObservation);
	         targets.add(curObservation[0]+curObservation[9]+curObservation[28]);
		}
          
	       /*
	        * LassoFitGenerator is initialized
	        */
	       LassoFitGenerator fitGenerator = new LassoFitGenerator();
	       try {
			fitGenerator.init(featuresCount, numObservations);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	       for (int i = 0; i < numObservations; i++) {
	               fitGenerator.setObservationValues(i, observations.get(i));
	               fitGenerator.setTarget(i, targets.get(i));
	       }
	       
	       /*
	        * Generate the Lasso fit. The -1 arguments means that
	        * there would be no limit on the maximum number of 
	        * features per model
	        */
	       LassoFit fit = fitGenerator.fit(-1);
	       
	       /*
	        * Print the generated fit
	        */
	       System.out.println(fit);
	       String filterStr="";
			 for (int i = 0; i <  fit.nonZeroWeights[fit.numberOfLambdas-1]; i++) {
					if(filterStr.length()>0)
						filterStr+=",";
					filterStr+=fit.indices[i]+1;
					System.out.println(fit.indices[i]+"---"+fit.getWeights(fit.numberOfLambdas-1)[fit.indices[i]]);
			}
			 System.out.println(filterStr);
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		testLiblinear();

	}

}
