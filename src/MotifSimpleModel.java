import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FilenameUtils;
import org.ejml.simple.SimpleMatrix;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.ling.LabeledWord;
import edu.stanford.nlp.sentiment.RNNOptions;
import edu.stanford.nlp.trees.Tree;


public class MotifSimpleModel implements Serializable {
	  /** 
	   * A random number generator - keeping it here lets us reproduce results
	   */
	  final Random rand;
	  static int threadNum=2;
	  public List<HashMap<String, Integer>> motif2FeatureId;
	  public List<Instances> trainFormats;
	  public List<Classifier> baseClassifiers;
	  public int numClasses;
	  public static List<String> ClassNames;
	  /**
	   * Will store various options specific to this model
	   */
	  final RNNOptions op;
	  
	  public MotifSimpleModel(RNNOptions op) {
		    this.op = op;
		    this.rand = new Random(op.randomSeed);
		    this.numClasses=op.numClasses;
		    this.baseClassifiers=new ArrayList<Classifier>(numClasses);
		    this.motif2FeatureId=new ArrayList<HashMap<String, Integer>>(numClasses);
		    this.trainFormats=new ArrayList<Instances>();
		   
		    if(threadNum>numClasses)
		    	threadNum=numClasses;
	  }
	
	  public static void setClassNames(String[] vals)
	  {
		  ClassNames=new ArrayList<String>();
		  for (int i = 0; i < vals.length; i++) {
			  ClassNames.add(vals[i]);
		}
	  }
	  public SimpleMatrix predict(Tree t)
	  {
		  SimpleMatrix classVal=new SimpleMatrix(numClasses, 1);
		  
		  for (int i = 0; i < baseClassifiers.size(); i++) {
			  	Classifier predictor = baseClassifiers.get(i);
				HashMap<String, Integer> motif_id = motif2FeatureId.get(i);
				double[] values = new double[motif_id.size()+1] ;
				for (LabeledWord leaf : t.labeledYield()) {
					String[] comps = leaf.toString().split("/");
					String motif_name=comps[0];
					if(!motif_id.containsKey(motif_name))
						continue;
					double score=Double.parseDouble(comps[1]);
					values[motif_id.get(motif_name)]=Math.max(values[motif_id.get(motif_name)], score);
				}
				Instance instance = new Instance(1, values);
				instance.setDataset(trainFormats.get(i));
				try {
					double cls=predictor.classifyInstance(instance);
					classVal.set(i, cls);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
		}
         //to do
		return classVal;
	  }
	  
	  
	  // to do
	  public void train(List<Tree> trees)
	  {
		  op.numClasses=MotifRNNCoreAnnotations.getGoldClass(trees.get(0)).numRows();
		  numClasses= op.numClasses;
		  motif2FeatureId.clear();
		  trainFormats.clear();
		   ExecutorService threadPool = Executors.newScheduledThreadPool(threadNum);
		   ArrayList<wekaLearningThread> threadlist=new ArrayList<wekaLearningThread>();
		  for (int i = 0; i < numClasses; i++) {
			//train n classifier
			  Instances traindata = DataSelector.Tree2Weka(trees, i);
			//build motif_name mapping
			  HashMap<String, Integer> motif_id=new HashMap<String, Integer>();
			 for (int j = 0; j < traindata.numAttributes(); j++) {
				 Attribute attr = traindata.attribute(j);
				 motif_id.put(attr.name(),j);
			} 
			 motif2FeatureId.add(motif_id);
			 trainFormats.add(traindata.stringFreeStructure());
			 wekaLearningThread weka_thread=new wekaLearningThread(traindata,i);
			 if(threadNum>1)
			 {
			 threadPool.execute(weka_thread);
			 }
			 else
			 {
			 weka_thread.run();
			 }
			 threadlist.add(weka_thread);
		}
		 
			try {
				baseClassifiers.clear();
			      threadPool.shutdown();
			      threadPool.awaitTermination(op.trainOptions.maxTrainTimeSeconds*7, TimeUnit.SECONDS);
			      for (int i = 0; i < threadlist.size(); i++) {
					baseClassifiers.add(threadlist.get(i).learner);
				}
			      
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

	  }
	  
	  
	  public void train(String filepath)
	  {
		  BufferedReader br;
			try {
				br = new BufferedReader(new FileReader(filepath));
				String line=br.readLine().substring(1);
				String[] comps = line.split(" ", 2);
				 op.numClasses=comps[0].split(",").length;
				 br.close();
			}
			catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		  numClasses= op.numClasses;
		  motif2FeatureId.clear();
		  trainFormats.clear();
		   ExecutorService threadPool = Executors.newScheduledThreadPool(threadNum);
		   ArrayList<wekaLearningThread> threadlist=new ArrayList<wekaLearningThread>();
		  for (int i = 0; i < numClasses; i++) {
			//train n classifier
			  Instances traindata = DataSelector.Tree2Weka(filepath, i);
			//build motif_name mapping
			  HashMap<String, Integer> motif_id=new HashMap<String, Integer>();
			 for (int j = 0; j < traindata.numAttributes(); j++) {
				 Attribute attr = traindata.attribute(j);
				 motif_id.put(attr.name(),j);
			} 
			 motif2FeatureId.add(motif_id);
			 trainFormats.add(traindata.stringFreeStructure());
			 wekaLearningThread weka_thread=new wekaLearningThread(traindata,i);
			 if(threadNum>1)
			 {
			 threadPool.execute(weka_thread);
			 }
			 else
			 {
			 weka_thread.run();
			 }
			 threadlist.add(weka_thread);
		}
		 
			try {
				baseClassifiers.clear();
			      threadPool.shutdown();
			      threadPool.awaitTermination(op.trainOptions.maxTrainTimeSeconds*7, TimeUnit.SECONDS);
			      for (int i = 0; i < threadlist.size(); i++) {
					baseClassifiers.add(threadlist.get(i).learner);
				}
			      
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

	  }
	  
	  
	  public void train_flatInput(String filepath)
	  {
		  BufferedReader br;
			try {
				HashSet<String>  classNames=new HashSet<String>();
				br = new BufferedReader(new FileReader(filepath));
				String line;
				while ((line = br.readLine()) != null) {
				String[] comps = line.split("\t");
					if(comps.length>4)
					{
						String[] cls_strs = comps[4].split(",");
						for (String c_str : cls_strs) {
							if(Utils.IsTargetHistoneMark(c_str))  // for debug only consider these marks
							classNames.add(c_str);
						}
					}
				}
				 
				 br.close();
				 setClassNames(classNames.toArray(new String[0]));
				 op.numClasses=classNames.size();
			}
			catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		 
		
			
		  numClasses= op.numClasses;
		  motif2FeatureId.clear();
		  trainFormats.clear();
		   ExecutorService threadPool = Executors.newScheduledThreadPool(threadNum);
		   ArrayList<wekaLearningThread> threadlist=new ArrayList<wekaLearningThread>();
		  for (int i = 0; i < numClasses; i++) {
			//train n classifier
			  System.err.println("Class "+i+" :" +ClassNames.get(i));
			  Instances traindata = DataSelector.FlatString2Weka(filepath, ClassNames.get(i));
			//build motif_name mapping
			  HashMap<String, Integer> motif_id=new HashMap<String, Integer>();
			 for (int j = 0; j < traindata.numAttributes(); j++) {
				 Attribute attr = traindata.attribute(j);
				 motif_id.put(attr.name(),j);
			} 
			 motif2FeatureId.add(motif_id);
			 trainFormats.add(traindata.stringFreeStructure());
			 wekaLearningThread weka_thread=new wekaLearningThread(traindata,i);
			 if(threadNum>1)
			 {
			 threadPool.execute(weka_thread);
			 }
			 else
			 {
			 weka_thread.run();
			 }
			 threadlist.add(weka_thread);
		}
		 
			try {
				baseClassifiers.clear();
			      threadPool.shutdown();
			      threadPool.awaitTermination(op.trainOptions.maxTrainTimeSeconds*7, TimeUnit.SECONDS);
			      for (int i = 0; i < threadlist.size(); i++) {
					baseClassifiers.add(threadlist.get(i).learner);
				}
			      
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			   System.err.println(ClassNames);
	  }
	  
	  
	  
	  
	  public void train_fastaInput(String filepath, String flankfile)
	  {
		  
		  HashSet<String>  classNames=new HashSet<String>();
		  String basename = FilenameUtils.getBaseName(filepath);
		  classNames.add(basename);
			setClassNames(classNames.toArray(new String[0]));
			op.numClasses=classNames.size();
		 
			threadNum=1;
		
			
		  numClasses= op.numClasses;
		  motif2FeatureId.clear();
		  trainFormats.clear();
		   ExecutorService threadPool = Executors.newScheduledThreadPool(threadNum);
		   ArrayList<wekaLearningThread> threadlist=new ArrayList<wekaLearningThread>();
		
		 
			try {
				//train a classifier
				Instances traindata=null;
				if(flankfile.length()==0)
				   traindata = DataSelector.Fasta2Weka(filepath);
				else
					traindata = DataSelector.Fasta2Weka(filepath,flankfile);
				//build motif_name mapping
				  HashMap<String, Integer> motif_id=new HashMap<String, Integer>();
				 for (int j = 0; j < traindata.numAttributes(); j++) {
					 Attribute attr = traindata.attribute(j);
					 motif_id.put(attr.name(),j);
				 }
				 motif2FeatureId.add(motif_id);
				 
				 
				 trainFormats.add(traindata.stringFreeStructure());
				 wekaLearningThread weka_thread=new wekaLearningThread(traindata,0);
				 if(threadNum>1)
				 {
				 threadPool.execute(weka_thread);
				 }
				 else
				 {
				 weka_thread.run();
				 }
				 threadlist.add(weka_thread);
			
				 
				baseClassifiers.clear();
			      threadPool.shutdown();
			      threadPool.awaitTermination(op.trainOptions.maxTrainTimeSeconds*7, TimeUnit.SECONDS);
			      for (int i = 0; i < threadlist.size(); i++) {
					baseClassifiers.add(threadlist.get(i).learner);
				}
			      
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

	  }
	  
	  
	  
	  public void predict_fastaInput(String filepath, String flankfile)
	  {

			try {
				//train a classifier
				  Instances testdata = null;
				  if(flankfile.length()>0)
					  testdata=DataSelector.Fasta2Weka(filepath,flankfile);
				  else
					  testdata=DataSelector.Fasta2Weka(filepath);
				  Evaluation eval1 = new Evaluation(testdata);
				Classifier model = baseClassifiers.get(0);
				double[]  predictions=eval1.evaluateModel(model, testdata);
				System.err.println(eval1.toSummaryString(true));
				System.err.println(eval1.toMatrixString());
				System.err.println("auc"+eval1.areaUnderROC(1));
				for (int i = 0; i < predictions.length; i++) {
					System.out.println(i+"\t"+model.distributionForInstance(testdata.instance(i))[1]);
				}
			   
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

	  }
	  
	  
	  
	  public void predict_flatInput(String filepath)
	  {
		  BufferedReader br;
		  HashSet<String>  classNames=new HashSet<String>();
		  LinkedList<String> lines=new LinkedList<String>();
			try {
				
				br = new BufferedReader(new FileReader(filepath));
				String line;
				while ((line = br.readLine()) != null) {
				String[] comps = line.split("\t");
					if(comps.length>4)
					{
						String[] cls_strs = comps[4].split(",");
						for (String c_str : cls_strs) {
							if(Utils.IsTargetHistoneMark(c_str))  // for debug only consider these marks
							classNames.add(c_str);
						}
					}
					lines.add(line);
				}
				 
				 br.close();
				 setClassNames(classNames.toArray(new String[0]));
				 op.numClasses=classNames.size();
			}
			catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			

			try {
				//train a classifier
				  Instances testdata = null;
				
				 Iterator<String> iter = classNames.iterator();
				 int mid=0;
				 String header="";
				ArrayList<Instances>  Testsets=new ArrayList<Instances>();
				 while(iter.hasNext())
				 {
					 String cls_str=iter.next();
					 header+="\t"+cls_str;
					  testdata=DataSelector.FlatString2Weka_knownFeatureMap(lines, cls_str,motif2FeatureId.get(mid));
						  Evaluation eval1 = new Evaluation(testdata);
						Classifier model = baseClassifiers.get(mid);
						double[]  predictions=eval1.evaluateModel(model, testdata);
						System.err.println(eval1.toSummaryString(true));
						System.err.println(eval1.toMatrixString());
						System.err.println("auc"+eval1.areaUnderROC(1));
						Testsets.add(testdata);
						mid+=1;
				 }
				 System.out.println(header);
				for (int i = 0; i < testdata.numInstances(); i++) {
					String outstr="";
					for (int j = 0; j < baseClassifiers.size(); j++) {
						outstr+="\t"+baseClassifiers.get(j).distributionForInstance(Testsets.get(j).instance(i))[1];
					}
					System.out.println(i+outstr+"\t"+lines.get(i).split("\t")[4]);//+"--"+testdata.instance(i).classValue()
				}
			   System.err.println(ClassNames);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

	  }
	  
	  public void saveSerialized(String path) {
//		    try {
//		      IOUtils.writeObjectToFile(this, path);
//		    } catch (IOException e) {
//		      throw new RuntimeIOException(e);
//		    }
			
	        try {
	       	 FileOutputStream fileOut =
	   		         new FileOutputStream(path);
	   		         ObjectOutputStream out =
	   		                            new ObjectOutputStream(fileOut);
				out.writeObject(this);
		         out.close();
		          fileOut.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		  }

	public static MotifSimpleModel loadSerialized(String path) {
//		    try {
//		      return IOUtils.readObjectFromURLOrClasspathOrFileSystem(path);
//		    } catch (IOException e) {
//		      throw new RuntimeIOException(e);
//		    } catch (ClassNotFoundException e) {
//		      throw new RuntimeIOException(e);
//		    }
		    
		    
			File f1=new File(path);
			if(f1.exists())
			{
				 FileInputStream fileIn;
				 try {
					fileIn = new FileInputStream(f1.getAbsolutePath());
					 ObjectInputStream in = new ObjectInputStream(fileIn);
					 MotifSimpleModel temp=(MotifSimpleModel)in.readObject();
					 fileIn.close();
					 return temp;
				} catch (Exception e) {
					// TODO Auto-generated catch block
					System.err.println(e.getMessage());
				} 
			
			}
			return null;
		  }

}
