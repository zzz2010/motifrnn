import java.io.PrintStream;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

import org.ejml.simple.SimpleMatrix;

import edu.stanford.nlp.sentiment.RNNOptions;
import edu.stanford.nlp.trees.Tree;


public class testSentimentTraining {

	static void test_realData()
	{
//		 System.err.close(); 
		  RNNOptions op = new RNNOptions();
		    op.randomSeed=0;
//		    String trainPath = "balance.fa.tree.5k";
		    String trainPath="singleEffect.txt";
//		    String trainPath="E004_enh_5k.tree";

		    boolean runGradientCheck = false;
		    boolean runTraining = true;

		    String modelPath ="./balance.model";
//		    System.err.println("Sentiment model options:\n" + op);
		   
		    
		    // read in the trees
		    List<Tree> trainingTrees = Utils.readTreesWithGoldLabels(trainPath);
		    Collections.shuffle(trainingTrees,new Random(op.randomSeed));
		    int maxTrains=Math.min(trainingTrees.size(), Utils.max_train_sample_size);
		    trainingTrees=trainingTrees.subList(0, maxTrains);
		    List<Tree> devTrees = trainingTrees.subList((int) (trainingTrees.size()-maxTrains*0.1), trainingTrees.size());
		    trainingTrees=trainingTrees.subList(0, (int) (trainingTrees.size()-maxTrains*0.1));
//		    devTrees=trainingTrees;
		    op.numClasses=MotifRNNCoreAnnotations.getGoldClass(trainingTrees.get(0)).numRows();
		    op.trainOptions.epochs=1000;
		    op.trainOptions.regClassification=0.0001;
		    op.trainOptions.regTransform=0.01;
		    op.trainOptions.batchSize=100;
		    op.trainOptions.learningRate=0.015;
		    op.trainOptions.regWordVector=0.00001;
		    op.numHid=1;
		    op.useTensors=false;
		    op.simplifiedModel=true;
		    
		    
		    System.err.println("Sentiment model options:\n" + op);
		    MotifSentimentModel model = new MotifSentimentModel(op, trainingTrees);
		    MotifSentimentTraining.neural_function=new TanhTransform();
		    

		    	MotifSentimentTraining.runGradientCheck(model, trainingTrees);
		    

		    if (runTraining) {
		    	model=MotifSentimentTraining.train_multiThread(model, modelPath, trainingTrees, devTrees,5);
		    	model.saveSerialized(modelPath);
		    }
		
		    System.out.println(model.unaryClassification.toString());
		    System.out.println(devTrees.get(1));
		    MotifSentimentTraining.Predict_SetLabel(model, devTrees.get(1));
		    System.out.println(devTrees.get(1));
		    
		    
		    Map<String, Float> word_imp = model.GetAllWordsImportance();
		    int i=0;
			  for (Entry<String, Float> feat : word_imp.entrySet()) {
				 System.out.println(feat.getKey()+" "+feat.getValue());
				 i++;
				 if(i>10)
					 break;
			}
			  
			  SimpleMatrix WordMat=new SimpleMatrix(word_imp.size(), op.numHid);
				  i =-1;
				 for (Entry<String, Float> feat : word_imp.entrySet()) {
					 SimpleMatrix vec = model.wordVectors.get(feat.getKey());
					 i+=1;
					 for (int j = 0; j < vec.getNumElements(); j++) {
						 WordMat.set(i,j, vec.get(j)); 
					}
				 }
				 SimpleMatrix PCs = WordMat.svd().getU();
				  i =-1;
					 for (Entry<String, Float> feat : word_imp.entrySet()) {
						 i+=1;
						 System.out.println(feat.getKey()+" "+feat.getValue()+" "+PCs.get(i, 0)+" "+PCs.get(i, 1));
					 }
		    
	}
	static void test_nonexchange_grammar()
	{
	    RNNOptions op = new RNNOptions();
	    op.randomSeed=0;
	    String trainPath = "nonExchangable_grammar_dataset.txt";

	    boolean runGradientCheck = false;
	    boolean runTraining = true;

	    String modelPath ="./test_nonexchange_grammar.model";

	   
	    
	    // read in the trees
	    List<Tree> trainingTrees = Utils.readTreesWithGoldLabels(trainPath);
	    Collections.shuffle(trainingTrees,new Random(op.randomSeed));
	    List<Tree> devTrees = trainingTrees.subList(trainingTrees.size()-100, trainingTrees.size());
	    trainingTrees=trainingTrees.subList(0, trainingTrees.size()-100);
	    // TODO: binarize the trees, then collapse the unary chains.
	    // Collapsed unary chains always have the label of the top node in
	    // the chain
	    // Note: the sentiment training data already has this done.
	    // However, when we handle trees given to us from the Stanford Parser,
	    // we will have to perform this step

	    // build an unitialized SentimentModel from the binary productions
	    op.numClasses=MotifRNNCoreAnnotations.getGoldClass(trainingTrees.get(0)).numRows();
	    op.trainOptions.epochs=1000;
	    op.trainOptions.regClassification=0;
	    op.trainOptions.regTransform=0;
	    op.trainOptions.batchSize=100;
	    op.trainOptions.learningRate=0.01;
	    op.trainOptions.regWordVector=0;
	    op.numHid=5;
	    op.useTensors=false;
	    op.simplifiedModel=true;
	    
//	    op.trainOptions.debugOutputEpochs=2;
	    System.err.println("Sentiment model options:\n" + op);
	    MotifSentimentModel model = new MotifSentimentModel(op, trainingTrees);
	    MotifSentimentTraining.neural_function=new TanhTransform();
	    

	    	MotifSentimentTraining.runGradientCheck(model, trainingTrees);
	    

	    if (runTraining) {
	    	MotifSentimentTraining.train(model, modelPath, trainingTrees, devTrees);
	    	model.saveSerialized(modelPath);
	    }
	
	    System.out.println(model.unaryClassification.toString());
	    System.out.println(devTrees.get(1));
	    MotifSentimentTraining.Predict_SetLabel(model, devTrees.get(1));
	    System.out.println(devTrees.get(1));
	    

	}
	
	
	static void test_exmapleFinder()
	{
	    RNNOptions op = new RNNOptions();

	    String trainPath = "nonExchangable_grammar_dataset.txt";

	    boolean runTraining = true;

	    String modelPath ="./test_nonexchange_grammar.model";
   
	    // read in the trees
	    List<Tree> trainingTrees = Utils.readTreesWithGoldLabels(trainPath);
	    List<Tree> devTrees = trainingTrees.subList(trainingTrees.size()-100, trainingTrees.size());
	    trainingTrees=trainingTrees.subList(0, trainingTrees.size()-100);

	    op.numClasses=MotifRNNCoreAnnotations.getGoldClass(trainingTrees.get(0)).numRows();
	    op.trainOptions.epochs=1000;
	    op.trainOptions.regClassification=0;
	    op.trainOptions.regTransform=0;
	    op.trainOptions.batchSize=100;
	    op.trainOptions.learningRate=0.01;
	    op.trainOptions.regWordVector=0;
	    op.numHid=5;
	    op.useTensors=false;
	    op.simplifiedModel=true;
	    op.randomSeed=0;
	    System.err.println("Sentiment model options:\n" + op);
	    MotifSentimentModel model = new MotifSentimentModel(op, trainingTrees);
	    MotifSentimentTraining.neural_function=new TanhTransform();
	    
	    

	    if (runTraining) {
	    	MotifSentimentTraining.train_multiThread(model, modelPath, trainingTrees, devTrees,2);
	    	model.saveSerialized(modelPath);
	    }
	
	    System.out.println(model.unaryClassification.toString());
	    System.out.println(devTrees.get(1));
	    MotifSentimentTraining.Predict_SetLabel(model, devTrees.get(1));
	    System.out.println(devTrees.get(1));
	    

	}
	
	static void test_multiThreads()
	{
	    RNNOptions op = new RNNOptions();

	    String trainPath = "summation_multoutput_dataset.txt";
	 

	    boolean runGradientCheck = false;
	    boolean runTraining = true;

	    String modelPath = "./test_multiThreads.model";

	   
	    List<Tree> gaintTrees=new LinkedList<Tree>();
	    // read in the trees
	    List<Tree> trainingTrees = Utils.readTreesWithGoldLabels(trainPath);
	    for (int i = 0; i < 1000; i++) {
	    	gaintTrees.addAll(trainingTrees);
		}
	    
	    List<Tree> devTrees = trainingTrees.subList(trainingTrees.size()-100, trainingTrees.size());
	    trainingTrees=gaintTrees;

	    // TODO: binarize the trees, then collapse the unary chains.
	    // Collapsed unary chains always have the label of the top node in
	    // the chain
	    // Note: the sentiment training data already has this done.
	    // However, when we handle trees given to us from the Stanford Parser,
	    // we will have to perform this step

	    // build an unitialized SentimentModel from the binary productions
	    op.numClasses=MotifRNNCoreAnnotations.getGoldClass(trainingTrees.get(0)).numRows();
//	    op.trainOptions.learningRate=0.01;
	    op.trainOptions.epochs=2;
	    op.trainOptions.regClassification=0;
	    op.trainOptions.regTransform=0;
	    op.trainOptions.batchSize=20000;
	    op.trainOptions.learningRate=0.01;
	    op.trainOptions.regWordVector=0;
	    op.numHid=10;
	    op.useTensors=true;
	    op.simplifiedModel=true;
	    op.randomSeed=0;
	    System.err.println("Sentiment model options:\n" + op);
	    MotifSentimentModel model = new MotifSentimentModel(op, trainingTrees);
	    MotifSentimentTraining.neural_function=new IdenTransform();
	    // TODO: need to handle unk rules somehow... at test time the tree
	    // structures might have something that we never saw at training
	    // time.  for example, we could put a threshold on all of the
	    // rules at training time and anything that doesn't meet that
	    // threshold goes into the unk.  perhaps we could also use some
	    // component of the accepted training rules to build up the "unk"
	    // parameter in case there are no rules that don't meet the
	    // threshold
	    
//	    if (runGradientCheck) {
//	    	MotifSentimentTraining.runGradientCheck(model, trainingTrees);
//	    }

	    if (runTraining) {
	    	MotifSentimentTraining.train_multiThread(model, modelPath, trainingTrees, devTrees,10);
//	    	MotifSentimentTraining.train(model, modelPath, trainingTrees, devTrees);
	    	model.saveSerialized(modelPath);
	    }
	    
	    System.out.println(model.unaryClassification.toString());
	    System.out.println(devTrees.get(1));
	    MotifSentimentTraining.Predict_SetLabel(model, devTrees.get(1));
	    System.out.println(devTrees.get(1));
	}
	
	
	static void test_linearCorrectness()
	{
	    RNNOptions op = new RNNOptions();

	    String trainPath = "summation_multoutput_dataset.txt";
	 

	    boolean runGradientCheck = false;
	    boolean runTraining = true;

	    String modelPath = "./test_linearCorrectness.model";

	   
	    
	    // read in the trees
	    List<Tree> trainingTrees = Utils.readTreesWithGoldLabels(trainPath);
	    List<Tree> devTrees = trainingTrees.subList(trainingTrees.size()-100, trainingTrees.size());
	    trainingTrees=trainingTrees.subList(0,trainingTrees.size()-100);

	    // TODO: binarize the trees, then collapse the unary chains.
	    // Collapsed unary chains always have the label of the top node in
	    // the chain
	    // Note: the sentiment training data already has this done.
	    // However, when we handle trees given to us from the Stanford Parser,
	    // we will have to perform this step

	    // build an unitialized SentimentModel from the binary productions
	    op.numClasses=MotifRNNCoreAnnotations.getGoldClass(trainingTrees.get(0)).numRows();
//	    op.trainOptions.learningRate=0.01;
	    op.trainOptions.epochs=10000;
	    op.trainOptions.regClassification=0;
	    op.trainOptions.regTransform=0;
	    op.trainOptions.batchSize=200;
	    op.trainOptions.learningRate=0.01;
	    op.trainOptions.regWordVector=0;
	    op.numHid=1;
	    op.useTensors=true;
	    op.simplifiedModel=true;
	    op.randomSeed=0;
	    System.err.println("Sentiment model options:\n" + op);
	    MotifSentimentModel model = new MotifSentimentModel(op, trainingTrees);
	    MotifSentimentTraining.neural_function=new IdenTransform();
	    // TODO: need to handle unk rules somehow... at test time the tree
	    // structures might have something that we never saw at training
	    // time.  for example, we could put a threshold on all of the
	    // rules at training time and anything that doesn't meet that
	    // threshold goes into the unk.  perhaps we could also use some
	    // component of the accepted training rules to build up the "unk"
	    // parameter in case there are no rules that don't meet the
	    // threshold
	    
//	    if (runGradientCheck) {
	    	MotifSentimentTraining.runGradientCheck(model, trainingTrees);
//	    }

	    if (runTraining) {
	    	MotifSentimentTraining.train(model, modelPath, trainingTrees, devTrees);
	    	model.saveSerialized(modelPath);
	    }
	    
	    System.out.println(model.unaryClassification.toString());
	    System.out.println(devTrees.get(1));
	    MotifSentimentTraining.Predict_SetLabel(model, devTrees.get(1));
	    System.out.println(devTrees.get(1));
	}
	
	
	static void test_leafweightCorrectness()
	{
	    RNNOptions op = new RNNOptions();

	    String trainPath = "summation_leafweight_dataset.txt";

	    boolean runGradientCheck = false;
	    boolean runTraining = true;

	    String modelPath = "./test_linearCorrectness.model";

	   
	    
	    // read in the trees
	    List<Tree> trainingTrees = Utils.readTreesWithGoldLabels(trainPath);
	    List<Tree> devTrees = trainingTrees.subList(trainingTrees.size()-100, trainingTrees.size());
	    trainingTrees=trainingTrees.subList(0,trainingTrees.size()-100);

	    // TODO: binarize the trees, then collapse the unary chains.
	    // Collapsed unary chains always have the label of the top node in
	    // the chain
	    // Note: the sentiment training data already has this done.
	    // However, when we handle trees given to us from the Stanford Parser,
	    // we will have to perform this step

	    // build an unitialized SentimentModel from the binary productions
	    op.numClasses=MotifRNNCoreAnnotations.getGoldClass(trainingTrees.get(0)).numRows();
//	    op.trainOptions.learningRate=0.01;
	    op.trainOptions.epochs=10000;
	    op.trainOptions.regClassification=0.01;
	    op.trainOptions.regTransform=0.001;
	    op.trainOptions.batchSize=200;
	    op.trainOptions.learningRate=0.01;
	    op.trainOptions.regWordVector=0;
	    op.numHid=1;
	    op.useTensors=false;
	    op.simplifiedModel=true;
	    op.randomSeed=0;
	    System.err.println("Sentiment model options:\n" + op);
	    MotifSentimentModel model = new MotifSentimentModel(op, trainingTrees);
	    MotifSentimentTraining.neural_function=new IdenTransform();
	    // TODO: need to handle unk rules somehow... at test time the tree
	    // structures might have something that we never saw at training
	    // time.  for example, we could put a threshold on all of the
	    // rules at training time and anything that doesn't meet that
	    // threshold goes into the unk.  perhaps we could also use some
	    // component of the accepted training rules to build up the "unk"
	    // parameter in case there are no rules that don't meet the
	    // threshold
	    
//	    if (runGradientCheck) {
//	    	MotifSentimentTraining.runGradientCheck(model, trainingTrees);
//	    }

	    if (runTraining) {
	    	MotifSentimentTraining.train(model, modelPath, trainingTrees, devTrees);
//	    	MotifSentimentTraining.train_multiThread(model, modelPath, trainingTrees, devTrees,3);
	    	model.saveSerialized(modelPath);
	    }
	    
	    System.out.println(model.unaryClassification.toString());
	    System.out.println(devTrees.get(1));
	    MotifSentimentTraining.Predict_SetLabel(model, devTrees.get(1));
	    System.out.println(devTrees.get(1));
	    

        
       

	}
	
	public static void test_PCA()
	{
		
		
		String modelPath="1_TssA_25_Quies_23_PromBiv_14_EnhA2.tree.model0.best.gz";//"1_TssA_25_Quies_23_PromBiv_14_EnhA2.tree.model0";
		MotifSentimentModel model = MotifSentimentModel.loadSerialized(modelPath);
		 Map<String, Float> word_imp = model.GetAllWordsImportance();
			//PCA reduce to 2D
			 SimpleMatrix WordMat=new SimpleMatrix(word_imp.size(), model.op.numHid);
			 int i =-1;
			 for (Entry<String, Float> feat : word_imp.entrySet()) {
				 SimpleMatrix vec = model.wordVectors.get(feat.getKey());
				 i+=1;
				 for (int j = 0; j < vec.getNumElements(); j++) {
					 WordMat.set(i,j, vec.get(j)); 
				}
			 }
			 SimpleMatrix PCs = WordMat.svd().getU();
			 i=-1;
			 for (Entry<String, Float> feat : word_imp.entrySet()) {
				 i+=1;
				 if(feat.getKey().contains("Pol2")||feat.getKey().contains("Ctcf"))
					 	System.out.println(feat.getKey()+" "+feat.getValue()+" "+PCs.get(i, 0)+" "+PCs.get(i, 1));
			 }
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
//		test_leafweightCorrectness();
//		test_linearCorrectness();
//		test_multiThreads();
//		test_exmapleFinder();
		test_nonexchange_grammar();
//		test_realData();
//		test_PCA();
	  }

}
