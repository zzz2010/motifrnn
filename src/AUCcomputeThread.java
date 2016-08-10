import java.util.List;

import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Timing;


public class AUCcomputeThread implements Runnable {

	int epoch;
	MotifSentimentModel model;
	MotifSentimentModel bestModel;
	double bestAccuracy;
	List<Tree> devTrees;
	Timing timing;
	String bestModelStr;
	
	
	public AUCcomputeThread(int epoch, MotifSentimentModel model,
			MotifSentimentModel bestModel, double bestAccuracy,
			List<Tree> devTrees, Timing timing, String bestModelStr) {
		super();
		this.epoch = epoch;
		this.model = model;
		this.bestModel = bestModel;
		this.bestAccuracy = bestAccuracy;
		this.devTrees = devTrees;
		this.timing = timing;
		this.bestModelStr = bestModelStr;
	}


	@Override
	public void run() {
		// TODO Auto-generated method stub
		long totalElapsed = timing.report();
        if ( epoch > 0 && epoch % model.op.trainOptions.debugOutputEpochs == 0) {
            double score = 0.0;
            System.err.println("======================================");
            System.out.println("Finished epoch " + epoch + "; total training time " + totalElapsed + " ms");
            if (devTrees != null) {
              MotifEvaluate eval = new MotifEvaluate(model);
              if(Utils.RNNdoubleTest)
              {
              	System.out.println("Double testing...........");
              	 eval.eval(devTrees.subList(0, devTrees.size()/2));
                   eval.printSummary();
                   String stateString="";
                   double avgAUC=eval.AUCs.elementSum()/eval.AUCs.getNumElements();
                   stateString+="UserTest Data AUC: "+avgAUC;
              	 	eval.eval(devTrees.subList( devTrees.size()/2+1,devTrees.size()));
                   eval.printSummary();
                   avgAUC=eval.AUCs.elementSum()/eval.AUCs.getNumElements();
                   stateString+="-------SplitTest Data AUC: "+avgAUC;
                   if(bestAccuracy<avgAUC)
                   {
                	   System.out.println("current best model :"+bestModelStr);
                	   bestAccuracy=avgAUC;
                	   bestModelStr=stateString;
                	   bestModel=new MotifSentimentModel(model.op, trainingTrees);
                	   bestModel.vectorToParams(model.paramsToVector());
                	   model.saveSerialized(modelPath+".best.gz");
                   }
                   
              }
              else
              {
              eval.eval(devTrees);
              eval.printSummary();
              }
//              score =1/eval.rootPredictionError; //eval.exactNodeAccuracy() * 100.0;
            } 
            model.saveSerialized(OneTrainingBatchThread.tempPath);
          }
        
      
      //check the investigate examples:
      if(Utils.findNonExchangableExample)
      {
	      List<List<Tree>> succExamples = ExampleFinder.predictableExamples(model, InvestigateExamples);
	      System.out.println("Number of Success Examples is "+succExamples.size());
	      if(succExamples.size()>0)
	      {
	    	  System.out.println("First Success Example:");
	    	  List<Tree> succ_example = succExamples.get(0);
	    	  for (int i = 0; i <succ_example.size(); i++) {
	    		  System.out.println("true:"+succ_example.get(i));
	    		  Tree pred = succ_example.get(i).deepCopy();
	    		  Predict_SetLabel(model, pred);
	    		  System.out.println("pred:"+pred);
			}
	      }
      }
	}

}
