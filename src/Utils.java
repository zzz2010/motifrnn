

import java.awt.Color;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.ejml.simple.SimpleMatrix;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.trees.MemoryTreebank;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CollectionUtils;
import edu.stanford.nlp.util.Filter;
import edu.stanford.nlp.util.Generics;

/**
 * In the Sentiment dataset converted to tree form, the labels on the
 * intermediate nodes are the sentiment scores and the leaves are the
 * text of the sentence.  This class provides routines to read a file
 * of those trees and attach the sentiment score as the GoldLabel
 * annotation.
 *
 * @author John Bauer
 */




public class Utils {
  private Utils() {} // static methods only

  static boolean findNonExchangableExample=false;
  static int max_train_sample_size=40000;
  static boolean noCV=false;
  static double region_length_threshold=0.2; //1000bp
  static double region_mappability_threshold=0.9; 
  static boolean useNoModificationRegionAsBG=false;
  static boolean excludeRepeat=false;
  static boolean sumScore=true; //otherwise use max score
  static boolean SQnormalization=true;
  static boolean displayOnlyMotif=true;
  static boolean studyGC_CpG_Repeat=false;   //only correct the length and mappability
  static int numTopFeatures=40;
  static int kmerLen=6;
  static boolean enableGapKmer=false;
  static int gapLen=6;
  static boolean RNNdoubleTest=true;
  static boolean CGcorrection=false;
  static boolean lengthMappabilityCorrection=false;
  static double regCoef=1;
  private final static char[] toChar = {'A','C','G','T'};

  
  /** converts a long representation of a kmer back to a string. 
   */
  public static String longToString(long l, int k) {
      return new String(longToChars(l,k));
  }
  
  
  public static String insertGaptoKmer(String kmer, int ngap)
  {
	  String ret=kmer.substring(0,kmer.length()/2);
	  for (int i = 0; i < ngap; i++) {
		  ret+="N";
	}
	  ret+=kmer.substring(kmer.length()/2);
	  return ret;
	  
  }
  
  public static int argMax(SimpleMatrix vector)
  {
	  double maxE=Double.MIN_VALUE;
	  int maxi=0;
	  for (int i = 0; i < vector.getNumElements(); i++) {
		if(vector.get(i)>maxE)
		{
			maxE=vector.get(i);
			maxi=i;
		}
	}
	  return maxi;
  }
  public static void scatterPlot(String savepath,List<double[]> redpoints ,List<double[]> bluepoints)
  {
	    XYSeriesCollection result = new XYSeriesCollection();
	    XYSeries series1 = new XYSeries("sample1");
	    for (int i = 0; i <redpoints.size(); i++) {
	    	series1.add(redpoints.get(i)[0], redpoints.get(i)[1]);
	    }
	    result.addSeries(series1);
	    XYSeries series2 = new XYSeries("sample2");
	    for (int i = 0; i <bluepoints.size(); i++) {
	    	series2.add(bluepoints.get(i)[0], bluepoints.get(i)[1]);
	    }
	    
	    result.addSeries(series2);
	    JFreeChart chart = ChartFactory.createScatterPlot("", "Dim1", "Dim2", result, PlotOrientation.VERTICAL, true, true, false);
	    chart.setBackgroundPaint(Color.white);
	    
	    XYPlot plot = (XYPlot) chart.getPlot();
	    NumberAxis domain = (NumberAxis) plot.getDomainAxis();
        domain.setRange(-8, 8);
        domain.setTickUnit(new NumberTickUnit(1));
        domain.setVerticalTickLabels(true);
        NumberAxis range = (NumberAxis) plot.getRangeAxis();
        range.setRange(-8, 8);
        range.setTickUnit(new NumberTickUnit(1));
        plot.setDomainGridlinePaint(Color.white);
        plot.setRangeGridlinePaint(Color.white);
	    try {
			ChartUtilities.saveChartAsPNG(new File(savepath), chart, 800, 600);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    
  }
  
  public static boolean IsTargetHistoneMark(String c_str)
  {
	  boolean flag=false;

	  if(studyGC_CpG_Repeat)
	  {
	  //six core marks
	  flag|=c_str.contains("H3K4me3");
	  flag|=c_str.contains("H3K4me1");
	  flag|=c_str.contains("H3K27");
	  flag|=c_str.contains("H3K36me");
	  flag|=c_str.contains("H3K9me");
	  }
	  else
	  {
//		  flag|=c_str.contains("H3K4me");
//		  flag|=c_str.contains("H3K27ac");
//		  flag|=c_str.contains("H3K36me");
//		  flag|=c_str.contains("H3K9ac");
//		  flag|=c_str.contains("H4K8ac");
//		  flag|=c_str.contains("H2A.Z");
//		  flag|=c_str.contains("H3K79me2");
		  
//		  flag|=c_str.contains("H3K23me2");
		  
		  return true;
		  
	  }
	  
	  return flag;
  }
  public static boolean checkCoFounder(String featureName)
  {
	  return (featureName.contains("content")||featureName.contains("CpG")||featureName.contains("Length")||featureName.contains("Mapp"));
  }
  public static  char[] longToChars(long l, int k) {
      char[] chars = new char[k];
      while (k-- > 0) {
          int b = (int)(l & 3);
          l >>= 2;
          chars[k] = toChar[b];
      }
      return chars;
  }

  public static long reverseComplement(long kmer,
          int k) {
	long out = 0;
	for (int i = 0; i < k; i++) {
	byte b = (byte)((kmer ^ 3) & 3);
	kmer >>= 2;
	out = (out << 2) | b;
	}
	return out;
}

  
  public static long charsToLong(char[] chars,int start, int end) {
      long out = 0;
      for (int i = start; i < end; i++) {
          out <<= 2;
          char newchar = chars[i];
          switch (newchar) {
          case 'A':
          case 'a':
              out += 0;
          break;
          case 'C':
          case 'c':
              out += 1;
          break;
          case 'G':
          case 'g':
              out += 2;
          break;
          case 'T':
          case 't':
              out += 3;
          break;

          default:
              break;
          }
      }
      return out;
  }
  
  public  static double round(double number,int digit)
  {
	  
	 
	  int tens=10^digit;
	  number = Math.round(number * tens);
	  number = number/tens;  
	  return number;
  }
  
  public static long charsToLong(char[] chars) {
      long out = 0;
      for (int i = 0; i < chars.length; i++) {
          out <<= 2;
          char newchar = chars[i];
          switch (newchar) {
          case 'A':
          case 'a':
              out += 0;
          break;
          case 'C':
          case 'c':
              out += 1;
          break;
          case 'G':
          case 'g':
              out += 2;
          break;
          case 'T':
          case 't':
              out += 3;
          break;

          default:
              break;
          }
      }
      return out;
  }

  
  public static void attachGoldLabels(Tree tree) {
    if (tree.isLeaf()) {
      return;
    }
    for (Tree child : tree.children()) {
      attachGoldLabels(child);
    }

    // In the sentiment data set, the node labels are simply the gold
    // class labels.  There are no categories encoded.
    MotifRNNCoreAnnotations.setGoldClass(tree, string2Matrix(tree.label().value()));
  }

  public static SimpleMatrix string2Matrix(String str)
  {
	  String[] comps = str.split(",");
	  SimpleMatrix mat=new SimpleMatrix(comps.length, 1);
	  for (int i = 0; i < comps.length; i++) {
		  mat.set(i, Double.valueOf(comps[i]));
	}
	  return mat;
  }
  
  public static String Vector2String(SimpleMatrix pred)
  {
	  String label = String.format("%.4g", pred.get(0)); 

	  for (int i = 1; i < pred.numRows(); i++) {
		  label+=","+ String.format("%.4g", pred.get(i)); 
	}
	  return label;
  }
  /**
   * Given a file name, reads in those trees and returns them as a List
   */
  public static List<Tree> readTreesWithGoldLabels(String path) {
    List<Tree> trees = Generics.newArrayList();
    MemoryTreebank treebank = new MemoryTreebank();
    treebank.loadPath(path, null);
    for (Tree tree : treebank) {
      attachGoldLabels(tree);
      trees.add(tree);
    }
    return trees;
  }

  static final Filter<Tree> UNKNOWN_ROOT_FILTER = new Filter<Tree>() {
    @Override
	public boolean accept(Tree tree) {
      int gold = RNNCoreAnnotations.getGoldClass(tree);
      return gold != -1;
    }
  };

  public static List<Tree> filterUnknownRoots(List<Tree> trees) {
    return CollectionUtils.filterAsList(trees, UNKNOWN_ROOT_FILTER);
  }

  public static String sentimentString(MotifSentimentModel model, int sentiment) {
    String[] classNames = model.op.classNames;
    if (sentiment < 0 || sentiment > classNames.length) {
      return "Unknown sentiment label " + sentiment;
    }
    return classNames[sentiment];
  }
  
  /**
   * Applies tanh to each of the entries in the matrix.  Returns a new matrix.
   */
  public static SimpleMatrix elementwiseApplyIden(SimpleMatrix input) {
    SimpleMatrix output =input;
    return output;
  }

  /**
   * Applies the derivative of tanh to each of the elements in the vector.  Returns a new matrix.
   */
  public static SimpleMatrix elementwiseApplyIdenDerivative(SimpleMatrix input) {
    SimpleMatrix output = new SimpleMatrix(input.numRows(), input.numCols());
    output.set(1.0);
    return output;
  }
  
  
  public static <K, V extends Comparable<? super V>> Map<K, V> 
  sortByValue( Map<K, V> map )
{
	  List<Map.Entry<K, V>> list =
	      new LinkedList<Map.Entry<K, V>>( map.entrySet() );
	  Collections.sort( list, new Comparator<Map.Entry<K, V>>()
	  {
	      public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 )
	      {
	          return (o1.getValue()).compareTo( o2.getValue() );
	      }
	  } );
	
	  Map<K, V> result = new LinkedHashMap<K, V>();
	  for (Map.Entry<K, V> entry : list)
	  {
	      result.put( entry.getKey(), entry.getValue() );
	  }
	  return result;
	}
  
  
  public static <K> Map<K, Float> 
  sortByAbsValue( Map<K, Float> map )
{
	  List<Map.Entry<K, Float>> list =
	      new LinkedList<Map.Entry<K, Float>>( map.entrySet() );
	  Collections.sort( list, new Comparator<Map.Entry<K,Float>>()
	  {
	      public int compare( Map.Entry<K, Float> o1, Map.Entry<K, Float> o2 )
	      {
	    	  
	    	 float  f1=(java.lang.Float) o1.getValue();
	    	 float  f2=(java.lang.Float) o2.getValue();
	    	 f1=Math.abs(f1);
	    	 f2=Math.abs(f2);
	          return -Float.compare(f1, f2);
	      }
	  } );
	
	  Map<K, Float> result = new LinkedHashMap<K, Float>();
	  for (Map.Entry<K, Float> entry : list)
	  {
	      result.put( entry.getKey(), entry.getValue() );
	  }
	  return result;
	}
  
  public static HashMap<String, Float> ReadFileToHashMap(String inputfile)
  {
	  HashMap<String, Float> map = new HashMap<String, Float>();
      BufferedReader in;
	try {
		in = new BufferedReader(new FileReader(inputfile));
	      String line = "";
	      while ((line = in.readLine()) != null) {
	          String parts[] = line.split("\t");
	          map.put(parts[0], Float.parseFloat(parts[1]));
	      }
	      in.close();
	      System.out.println(map.toString());
	      
	      
	} catch (FileNotFoundException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	} catch (IOException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
	return map;

  }
  
}



