import java.util.List;

import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Timing;


public class OneTrainingBatchThread implements Runnable {

	MotifSentimentModel model;
	List<Tree> trainingBatch;
	double[] sumGradSquare;
	Timing timing;
	long maxTrainTimeMillis;
	int epoch;
	static String tempPath;
	int batch;
	List<Tree> devTrees;
	public OneTrainingBatchThread(MotifSentimentModel model, List<Tree> trainingBatch, double[] sumGradSquare,int epoch,int batch,List<Tree> devTrees,Timing timing,long maxTrainTimeMillis)
	{
		this.model=model;
		this.trainingBatch=trainingBatch;
		this.sumGradSquare=sumGradSquare;
		this.epoch=epoch;
		this.batch=batch;
		this.devTrees=devTrees;
		this.timing=timing;
		this.maxTrainTimeMillis=maxTrainTimeMillis;
	}
	@Override
	public void run() {
		// TODO Auto-generated method stub
		
		MotifSentimentTraining.executeOneTrainingBatch(model, trainingBatch, sumGradSquare);

        long totalElapsed = timing.report();
//        System.err.println("Finished epoch " + epoch + " batch " + batch + "; total training time " + totalElapsed + " ms");

        if (maxTrainTimeMillis > 0 && totalElapsed > maxTrainTimeMillis) {
          // no need to debug output, we're done now
            System.err.println("======================================");
            System.err.println("Epoch " + epoch + " batch " + batch);
            System.err.println("Finished epoch " + epoch + " batch " + batch + "; total training time " + totalElapsed/1000 + " secs");
          return;
        }

//        if (batch == 0 && epoch > 0 && epoch % model.op.trainOptions.debugOutputEpochs == 0) {
//          double score = 0.0;
//          System.err.println("======================================");
//          System.err.println("Epoch " + epoch + " batch " + batch);
//          System.out.println("Finished epoch " + epoch + " batch " + batch + "; total training time " + totalElapsed + " ms");
//          if (devTrees != null) {
//            MotifEvaluate eval = new MotifEvaluate(model);
//            if(Utils.RNNdoubleTest)
//            {
//            	System.out.println("Double testing...........");
//            	 eval.eval(devTrees.subList(0, devTrees.size()/2));
//                 eval.printSummary();
//            	 eval.eval(devTrees.subList( devTrees.size()/2+1,devTrees.size()));
//                 eval.printSummary();
//                 
//            }
//            else
//            {
//            eval.eval(devTrees);
//            eval.printSummary();
//            }
////            score =1/eval.rootPredictionError; //eval.exactNodeAccuracy() * 100.0;
//          } 
//          model.saveSerialized(tempPath);
//        }

	}

}
