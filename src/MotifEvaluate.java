

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

import org.ejml.simple.SimpleMatrix;

import auc.AUCCalculator;
import auc.Confusion;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentUtils;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.StringUtils;

public class MotifEvaluate {
  final MotifSentimentCostAndGradient cag;
  final MotifSentimentModel model;

  final int[][] equivalenceClasses;
  final String[] equivalenceClassNames;
  
  SimpleMatrix labelsCorrect;
  SimpleMatrix labelsIncorrect;

  // the matrix will be [gold][predicted]
  int[][] labelConfusion;

  int rootLabelsCorrect;
  int rootLabelsIncorrect;
  SimpleMatrix rootPredictionError;
  int[][] rootLabelConfusion;
  SimpleMatrix AUCs;
  IntCounter<Integer> lengthLabelsCorrect;
  IntCounter<Integer> lengthLabelsIncorrect;

  private static final NumberFormat NF = new DecimalFormat("0.000000");

  public MotifEvaluate(MotifSentimentModel model) {
    this.model = model;
    this.cag = new MotifSentimentCostAndGradient(model, null,MotifSentimentTraining.neural_function);
    this.equivalenceClasses = model.op.equivalenceClasses;
    this.equivalenceClassNames = model.op.equivalenceClassNames;

    reset();
  }

  public void reset() {

    labelConfusion = new int[model.op.numClasses][model.op.numClasses];
    AUCs=new SimpleMatrix(model.numClasses, 1);
    rootLabelsCorrect = 0;
    rootLabelsIncorrect = 0;
    rootLabelConfusion = new int[model.op.numClasses][model.op.numClasses];
 
    lengthLabelsCorrect = new IntCounter<Integer>();
    lengthLabelsIncorrect = new IntCounter<Integer>();
  }

  public void eval(List<Tree> trees) {
	  rootPredictionError=new SimpleMatrix(model.numClasses, 1);
	  labelsCorrect=new SimpleMatrix(model.numClasses, 1);
	  labelsIncorrect=new SimpleMatrix(model.numClasses, 1);
    for (Tree tree : trees) {
      eval(tree);
    }
    rootPredictionError=rootPredictionError.scale(1.0/trees.size());
    
    AUCs=new SimpleMatrix(model.numClasses, 1);
    //evaluate AUC for different class
    
    Random rand=new Random(model.op.randomSeed)  ;
    ArrayList<TreeMap<Double,Integer>> Sorted_labels_List=new ArrayList<TreeMap<Double,Integer>>(model.op.numClasses);
	for (int i = 0; i < model.numClasses; i++) {
		Sorted_labels_List.add(new TreeMap<Double,Integer>());
	}
	double[] goldLabelMean=new double[model.numClasses]; //use for binarize
    for (Tree tree : trees) {
        SimpleMatrix gold = MotifRNNCoreAnnotations.getGoldClass(tree);
	    SimpleMatrix predicted = MotifRNNCoreAnnotations.getPredictions(tree);
	    for (int i = 0; i < model.numClasses; i++) {
	    	Sorted_labels_List.get(i).put(predicted.get(i)+(rand.nextDouble()-0.5)/100000, (int) gold.get(i));
	    	goldLabelMean[i]+=(int)gold.get(i);
	    }
      }
    
    for (int i = 0; i <model.numClasses; i++) {
    	goldLabelMean[i]/=trees.size();
    	TreeMap<Double, Integer> Sorted_labels = Sorted_labels_List.get(i);
	   	double[]  scores=new double[Sorted_labels.size()];
	    int[]  labels=new int[Sorted_labels.size()];
	  	 int ii=0;
	  	 for(Double key:Sorted_labels.descendingKeySet())
	  	 {
	  		 if(Sorted_labels.get(key)>goldLabelMean[i])
	  			labels[ii]=1;
	  		 scores[ii]=key;
	  		        ii++;
	  	 }
		 
	  	Confusion AUCcalc=AUCCalculator.readArrays(labels, scores);	
	  	 double AUCscore=AUCcalc.calculateAUCROC();
	  	AUCs.set(i, AUCscore);
    }
    
    
    
  }

  public SimpleMatrix eval(Tree tree) {
    cag.forwardPropagateTree(tree);

//    countTree(tree);
    return countRoot(tree);
//    countLengthAccuracy(tree);
  }

  private int countLengthAccuracy(Tree tree) {
    if (tree.isLeaf()) {
      return 0;
    }
    Integer gold = RNNCoreAnnotations.getGoldClass(tree);
    Integer predicted = RNNCoreAnnotations.getPredictedClass(tree);
    int length;
    if (tree.isPreTerminal()) {
      length = 1;
    } else {
      length = 0;
      for (Tree child : tree.children()) {
        length += countLengthAccuracy(child);
      }
    }
    if (gold >= 0) {
      if (gold.equals(predicted)) {
        lengthLabelsCorrect.incrementCount(length);
      } else {
        lengthLabelsIncorrect.incrementCount(length);
      }
    }
    return length;
  }

//  private void countTree(Tree tree) {
//    if (tree.isLeaf()) {
//      return;
//    }
//    for (Tree child : tree.children()) {
//      countTree(child);
//    }
//    Integer gold = RNNCoreAnnotations.getGoldClass(tree);
//    Integer predicted = RNNCoreAnnotations.getPredictedClass(tree);
//    if (gold >= 0) {
//      if (gold.equals(predicted)) {
//        labelsCorrect++;
//      } else {
//        labelsIncorrect++;
//      }
//      labelConfusion[gold][predicted]++;
//    }
//  }

  private SimpleMatrix countRoot(Tree tree) {
    SimpleMatrix gold = MotifRNNCoreAnnotations.getGoldClass(tree);
    SimpleMatrix predicted = MotifRNNCoreAnnotations.getPredictions(tree);
//    double sqErr = predicted.minus(gold).normF();
     SimpleMatrix err = predicted.minus(gold);
     double cutoff=0;
     for (int i = 0; i < err.numRows(); i++) {
    	 if(Math.abs(gold.get(i))>cutoff)
    	 {
    		 double correct = labelsCorrect.get(i);
    		 double wrong = labelsIncorrect.get(i);
    		 if(Math.round(predicted.get(i))==gold.get(i))
    			 labelsCorrect.set(i, correct+1);
    		 else
    			 labelsIncorrect.set(i, wrong+1);
    	 }
    	 rootPredictionError.set(i, rootPredictionError.get(i)+Math.abs(err.get(i))); //here use normF instead of sqErr, because easy to judge
	}
    
    return err;
  }

  public SimpleMatrix exactNodeAccuracy() {
	  SimpleMatrix accuracy=new SimpleMatrix(labelsCorrect.numRows(), labelsCorrect.numCols());
	  for (int i = 0; i < labelsCorrect.numRows(); i++) {
		for (int j = 0; j < labelsCorrect.numCols(); j++) {
			accuracy.set(i, j, labelsCorrect.get(i, j)/(labelsCorrect.get(i, j)+labelsIncorrect.get(i, j)));
		}
	}
    return accuracy;
  }

  public double exactRootAccuracy() {
    return (double) rootLabelsCorrect / ((double) (rootLabelsCorrect + rootLabelsIncorrect));
  }

  public Counter<Integer> lengthAccuracies() {
    Set<Integer> keys = Generics.newHashSet();
    keys.addAll(lengthLabelsCorrect.keySet());
    keys.addAll(lengthLabelsIncorrect.keySet());

    Counter<Integer> results = new ClassicCounter<Integer>();
    for (Integer key : keys) {
      results.setCount(key, lengthLabelsCorrect.getCount(key) / (lengthLabelsCorrect.getCount(key) + lengthLabelsIncorrect.getCount(key)));
    }
    return results;
  }

  public void printLengthAccuracies() {
    Counter<Integer> accuracies = lengthAccuracies();
    Set<Integer> keys = Generics.newTreeSet();
    keys.addAll(accuracies.keySet());
    System.err.println("Label accuracy at various lengths:");
    for (Integer key : keys) {
      System.err.println(StringUtils.padLeft(Integer.toString(key), 4) + ": " + NF.format(accuracies.getCount(key)));
    }
  }

  private static void printConfusionMatrix(String name, int[][] confusion) {
    System.err.println(name + " confusion matrix: rows are gold label, columns predicted label");
    for (int i = 0; i < confusion.length; ++i) {
      for (int j = 0; j < confusion[i].length; ++j) {
        System.err.print(StringUtils.padLeft(confusion[i][j], 10));
      }
      System.err.println();
    }
  }

  private static double[] approxAccuracy(int[][] confusion, int[][] classes) {
    int[] correct = new int[classes.length];
    int[] incorrect = new int[classes.length];
    double[] results = new double[classes.length];
    for (int i = 0; i < classes.length; ++i) {
      for (int j = 0; j < classes[i].length; ++j) {
        for (int k = 0; k < classes[i].length; ++k) {
          correct[i] += confusion[classes[i][j]][classes[i][k]];
        }
      }
      for (int other = 0; other < classes.length; ++other) {
        if (other == i) {
          continue;
        }
        for (int j = 0; j < classes[i].length; ++j) {
          for (int k = 0; k < classes[other].length; ++k) {
            incorrect[i] += confusion[classes[i][j]][classes[other][k]];
          }
        }
      }
      results[i] = ((double) correct[i]) / ((double) (correct[i] + incorrect[i]));
    }
    return results;
  }

  private static double approxCombinedAccuracy(int[][] confusion, int[][] classes) {
    int correct = 0;
    int incorrect = 0;
    for (int i = 0; i < classes.length; ++i) {
      for (int j = 0; j < classes[i].length; ++j) {
        for (int k = 0; k < classes[i].length; ++k) {
          correct += confusion[classes[i][j]][classes[i][k]];
        }
      }
      for (int other = 0; other < classes.length; ++other) {
        if (other == i) {
          continue;
        }
        for (int j = 0; j < classes[i].length; ++j) {
          for (int k = 0; k < classes[other].length; ++k) {
            incorrect += confusion[classes[i][j]][classes[other][k]];
          }
        }
      }
    }
    return ((double) correct) / ((double) (correct + incorrect));
  }

  public void printSummary() {
    System.out.println("EVALUATION SUMMARY");
    System.out.println("Tested Avg Error:" + Utils.Vector2String(rootPredictionError));
//    System.err.println("Tested " + (labelsCorrect + labelsIncorrect) + " labels");
//    System.err.println("  " + labelsCorrect + " correct");
//    System.err.println("  " + labelsIncorrect + " incorrect");
    System.out.println("Test Avg Accuracy: " + Utils.Vector2String(exactNodeAccuracy()));
    
    System.out.println("Test AUCs: " + Utils.Vector2String(AUCs));
//    System.err.println("Tested " + (rootLabelsCorrect + rootLabelsIncorrect) + " roots");
//    System.err.println("  " + rootLabelsCorrect + " correct");
//    System.err.println("  " + rootLabelsIncorrect + " incorrect");
//    System.err.println("  " + NF.format(exactRootAccuracy()) + " accuracy");
//
//    printConfusionMatrix("Label", labelConfusion);
//    printConfusionMatrix("Root label", rootLabelConfusion);
//
//    if (equivalenceClasses != null && equivalenceClassNames != null) {
//      double[] approxLabelAccuracy = approxAccuracy(labelConfusion, equivalenceClasses);
//      for (int i = 0; i < equivalenceClassNames.length; ++i) {
//        System.err.println("Approximate " + equivalenceClassNames[i] + " label accuracy: " + NF.format(approxLabelAccuracy[i]));
//      }
//      System.err.println("Combined approximate label accuracy: " + NF.format(approxCombinedAccuracy(labelConfusion, equivalenceClasses)));
//      
//      double[] approxRootLabelAccuracy = approxAccuracy(rootLabelConfusion, equivalenceClasses);
//      for (int i = 0; i < equivalenceClassNames.length; ++i) {
//        System.err.println("Approximate " + equivalenceClassNames[i] + " root label accuracy: " + NF.format(approxRootLabelAccuracy[i]));
//      }
//      System.err.println("Combined approximate root label accuracy: " + NF.format(approxCombinedAccuracy(rootLabelConfusion, equivalenceClasses)));
//    }

    //printLengthAccuracies();
  }

  /**
   * Expected arguments are <code> -model model -treebank treebank </code> <br>
   *
   * For example <br>
   * <code> 
   *  java edu.stanford.nlp.sentiment.MotifEvaluate 
   *   edu/stanford/nlp/models/sentiment/sentiment.ser.gz 
   *   /u/nlp/data/sentiment/trees/dev.txt
   * </code>
   */
  public static void main(String[] args) {
    String modelPath = null;
    String treePath = null;
    boolean filterUnknown = false;

    for (int argIndex = 0; argIndex < args.length; ) {
      if (args[argIndex].equalsIgnoreCase("-model")) {
        modelPath = args[argIndex + 1];
        argIndex += 2;
      } else if (args[argIndex].equalsIgnoreCase("-treebank")) {
        treePath = args[argIndex + 1];
        argIndex += 2;
      } else if (args[argIndex].equalsIgnoreCase("-filterUnknown")) {
        filterUnknown = true;
        argIndex++;
      } else {
        System.err.println("Unknown argument " + args[argIndex]);
        System.exit(2);
      }
    }

    List<Tree> trees = SentimentUtils.readTreesWithGoldLabels(treePath);
    if (filterUnknown) {
      trees = SentimentUtils.filterUnknownRoots(trees);
    }
    MotifSentimentModel model = MotifSentimentModel.loadSerialized(modelPath);

    MotifEvaluate eval = new MotifEvaluate(model);
    eval.eval(trees);
    eval.printSummary();
  }
}
