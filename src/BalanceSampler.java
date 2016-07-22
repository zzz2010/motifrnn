import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.EmpiricalDistribution;

import de.bwaldvogel.liblinear.SolverType;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import edu.stanford.nlp.util.ArrayUtils;
import edu.wlu.cs.levy.CG.KDTree;
import edu.wlu.cs.levy.CG.KeyDuplicateException;
import edu.wlu.cs.levy.CG.KeyMissingException;
import edu.wlu.cs.levy.CG.KeySizeException;


public class BalanceSampler {
	static double jitter=0.000000000000001;
	static int numbin=100;
	static int numFeat=2;
	static int maxSampleSize=Integer.MAX_VALUE;
	static boolean DemoFlag=false;
	static double EPS=0.01;
	
	
	public static ArrayList<ArrayList<Integer>> getBalanceSetIDs_LinearOut(List<double[]> keyList1,List<double[]> keyList2 )
	{
		numFeat=keyList1.get(0).length; //resure the numfeat
		int featureNum=numFeat;
		FastVector attrList=new FastVector(featureNum+1);
		for (int i = 0; i < featureNum; i++) {
			attrList.addElement(new Attribute("attr"+i));
		}
		
		FastVector Tvalues = new FastVector();
		Tvalues.addElement("negative");
		Tvalues.addElement("positive");
		Attribute label = new Attribute("target_value",Tvalues);
		attrList.addElement(label);
		
		 Instances weka_traindata=new Instances("testLiblinear", attrList,keyList1.size()+keyList2.size());
		weka_traindata.setClassIndex(featureNum);
		
		for (int i = 0; i < keyList1.size(); i++) {
			 double[] curObservation = new double[featureNum+1];
			 double[] tmp=keyList1.get(i);
			 for (int j = 0; j < tmp.length; j++) {
				 curObservation[j]=tmp[j];
			}
			curObservation[curObservation.length-1]=1;
			Instance instance = new Instance(1, curObservation);
			 weka_traindata.add(instance);
		}
		
		for (int i = 0; i < keyList2.size(); i++) {
			 double[] curObservation = new double[featureNum+1];
			 double[] tmp=keyList2.get(i);
			 for (int j = 0; j < tmp.length; j++) {
				 curObservation[j]=tmp[j];
			}
			Instance instance = new Instance(1, curObservation);
			 weka_traindata.add(instance);
		}
		LibLINEAR2 model=new  LibLINEAR2();
		model.setSVMType(new SelectedTag(SolverType.L2R_LR_DUAL.getId(),LibLINEAR2.TAGS_SVMTYPE));
		model.setProbabilityEstimates(true);
//		LibSVM model=new LibSVM();
		try {
			weka.filters.supervised.instance.Resample filter=new Resample();
			filter.setInputFormat(weka_traindata);
			filter.setBiasToUniformClass(1);
			filter.setNoReplacement(true);
			filter.setSampleSizePercent(100*Math.min(1, 20000.1/weka_traindata.numInstances()));
			Instances resampleSets = Filter.useFilter(weka_traindata, filter);
			model.buildClassifier(resampleSets);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		ArrayList<double[]> newkeyList1=new ArrayList<double[]>();
		ArrayList<double[]> newkeyList2=new ArrayList<double[]>();
		for (int i = 0; i < weka_traindata.numInstances(); i++) {
			Instance inst=weka_traindata.instance(i);
			double[] pred=null;
			try {
				pred = model.distributionForInstance(inst);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			if(inst.classValue()==1.0)
			{
				
				newkeyList1.add(pred);
			}
			else
			{
				newkeyList2.add(pred);
			}
		}
		
		return getBalanceSetIDs_jointDist(newkeyList1,newkeyList2);
		
	}
	
	public static ArrayList<ArrayList<Integer>> getBalanceSetIDs_jointDistIter(List<double[]> keyList1,List<double[]> keyList2 )
	{
		numFeat=keyList1.get(0).length; //resure the numfeat
		ArrayList<ArrayList<Integer>> retIDs=new ArrayList<ArrayList<Integer>>();
		for (int i = 0; i <2; i++) {
			retIDs.add(new ArrayList<Integer>());
		}
		for (int i = 0; i < keyList1.size(); i++) {
			retIDs.get(0).add(i);
		}
		for (int i = 0; i < keyList2.size(); i++) {
			retIDs.get(1).add(i);
		}
		
		double maxErr=0;
		int lastsize1=keyList1.size();
		int lastsize2=keyList2.size();
		int iterNum=0;
		do
		{
			iterNum+=1;
			if(maxErr>0)
			{

				
				//random delete 5% of larger keylist
				if(lastsize2>lastsize1)
				{
					Collections.shuffle(keyList2, new Random(1));
					Collections.shuffle(retIDs.get(1), new Random(1));
					keyList2=new ArrayList<double[]>(keyList2.subList(0, (int) (keyList2.size()*0.95)));
					retIDs.set(1, new ArrayList<Integer>(retIDs.get(1).subList(0, keyList2.size())));
				}
				else
				{
					Collections.shuffle(keyList1, new Random(1));
					Collections.shuffle(retIDs.get(0), new Random(1));
					keyList1=new ArrayList<double[]>(keyList1.subList(0, (int) (keyList1.size()*0.95)));
					retIDs.set(0, new ArrayList<Integer>( retIDs.get(0).subList(0, keyList1.size())));
				}
			}
			maxErr=0;
			ArrayList<ArrayList<Integer>> currSelIDs = getBalanceSetIDs_jointDist(keyList1,keyList2);
			//check the err
			double[] mean1=new double[numFeat];
			double[] mean2=new double[numFeat];
			for (int i = 0; i < currSelIDs.get(0).size(); i++) {
				double[] tmp=keyList1.get(currSelIDs.get(0).get(i));
				for (int j = 0; j < numFeat; j++) {
					mean1[j]+=tmp[j];
				}
				
			}
			for (int i = 0; i < currSelIDs.get(1).size(); i++) {
				double[] tmp=keyList2.get(currSelIDs.get(1).get(i));
				for (int j = 0; j < numFeat; j++) {
					mean2[j]+=tmp[j];
				}
				
			}
			for (int j = 0; j < numFeat; j++)
			{
				mean1[j]/=currSelIDs.get(0).size();
				mean2[j]/=currSelIDs.get(1).size();
				double err=Math.abs(mean1[j]-mean2[j]);
				if(err>maxErr)
					maxErr=err;
			}
			//filter by current selection
			ArrayList<ArrayList<Integer>> retIDs2=new ArrayList<ArrayList<Integer>>();
			for (int i = 0; i <2; i++) {
				retIDs2.add(new ArrayList<Integer>());
			}
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < currSelIDs.get(i).size(); j++) {
					retIDs2.get(i).add(retIDs.get(i).get(currSelIDs.get(i).get(j)));
				}
			}
			retIDs=retIDs2;
			if(DemoFlag)
			{
				Utils.scatterPlot(iterNum+".png", keyList1, keyList2);
			}
			if(maxErr>EPS)
			{
				//reconstruct Keylist
				 lastsize1=keyList1.size();
				 lastsize2=keyList2.size();
				ArrayList<double[]> newK1=new ArrayList<double[]>();
				for (int i = 0; i <currSelIDs.get(0).size(); i++) {
					newK1.add(keyList1.get(currSelIDs.get(0).get(i)));
				}
				keyList1=newK1;
				ArrayList<double[]> newK2=new ArrayList<double[]>();
				for (int i = 0; i <currSelIDs.get(1).size(); i++) {
					newK2.add(keyList2.get(currSelIDs.get(1).get(i)));
				}
				keyList2=newK2;
			}
		}while(maxErr>EPS&&(lastsize1+lastsize2)>1000); //otherwise maybe too few example to learn
		return retIDs;
	}
	
	
	public static double calculateDistance(double[] array1, double[] array2)
    {
        double Sum = 0.0;
        for(int i=0;i<array1.length;i++) {
           Sum = Sum + Math.pow((array1[i]-array2[i]),2.0);
        }
        return Math.sqrt(Sum);
    }
	//this function directly take the joint feature to query, instead of independently generate each dimension from empirical
	// this method will provide the optimal pairing, and remove the points can not find pair within given max_allow_dist
		public static ArrayList<ArrayList<Integer>> getBalanceSetIDs_jointDist_bound(List<double[]> keyList1,List<double[]> keyList2, double max_allow_dist )
		{
			numFeat=keyList1.get(0).length; //resure the numfeat
			ArrayList<ArrayList<Integer>> retIDs=new ArrayList<ArrayList<Integer>>();
			if(keyList2.size()<keyList1.size())
			{
				
				ArrayList<ArrayList<Integer>> revlist=getBalanceSetIDs_jointDist_bound(keyList2,keyList1,max_allow_dist);
				retIDs.add(revlist.get(1));
				retIDs.add(revlist.get(0));
				return retIDs;
			}
			

			for (int i = 0; i <2; i++) {
				retIDs.add(new ArrayList<Integer>());
			}
			
			ArrayList<ArrayList<Double>> FeatVals1=new ArrayList<ArrayList<Double>>(numFeat); //use to produce empirical distribution
			ArrayList<ArrayList<Double>> FeatVals2=new ArrayList<ArrayList<Double>>(numFeat);
			ArrayList<ArrayList<Double>> FeatVals_resample=new ArrayList<ArrayList<Double>>(numFeat);
			ArrayList<ArrayList<Double>> FeatVals_resample0=new ArrayList<ArrayList<Double>>(numFeat);
			KDTree<Integer> KDTree1=new KDTree<Integer>(numFeat);
			KDTree<Integer> KDTree2=new KDTree<Integer>(numFeat);
		
			for (int i = 0; i < numFeat; i++) {
				FeatVals1.add(new ArrayList<Double>());
				FeatVals2.add(new ArrayList<Double>());
				FeatVals_resample.add(new ArrayList<Double>());
				FeatVals_resample0.add(new ArrayList<Double>());
			}
			Random rand=new Random(10);
			for (int i = 0; i < keyList1.size(); i++) {
				double[] feat=keyList1.get(i);
				for (int j= 0; j < numFeat; j++) {
					 feat[j] += (rand.nextDouble()-0.5)*i*jitter;
					 FeatVals1.get(j).add(feat[j] );
				}
			}
			
			for (int i = 0; i < keyList2.size(); i++) {
				double[] feat=keyList2.get(i);
				for (int j= 0; j < numFeat; j++) {
					 feat[j] += (rand.nextDouble()-0.5)*i*jitter;
					 FeatVals2.get(j).add(feat[j] );
				}
				try {
					keyList2.set(i, feat);
					if(keyList2.size()>keyList1.size())
					KDTree2.insert(feat,KDTree2.size());
				} catch (KeySizeException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (KeyDuplicateException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			
			ArrayList<EmpiricalDistribution> featDistLists=new ArrayList<EmpiricalDistribution>();
			for (int i = 0; i < FeatVals1.size(); i++) {
				EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numbin);
				CGdistribution.load(ArrayUtils.toPrimitive(FeatVals1.get(i).toArray(new Double[0])));
				featDistLists.add(CGdistribution);
			}
			
			int numQuerys = Math.min(maxSampleSize,  keyList1.size());
			ArrayList<Integer> QueryIds=new ArrayList<Integer>(numQuerys);
			for (int j = 0; j < numQuerys; j++) 
			{
				QueryIds.add(j);
			}
			
			while(true)
			{
				
				for (int i = 0; i < numFeat; i++) {

					FeatVals_resample.get(i).clear();
					FeatVals_resample0.get(i).clear();
				}
			
				
				TreeMap<Integer,LinkedList<Integer>> conflictPairs=new TreeMap<Integer,LinkedList<Integer>>();
				for(int j : QueryIds) {
					double [] query=keyList1.get(j);

					int sel_bgID;
					try {
						sel_bgID = KDTree2.nearest(query);
						//KDTree2.delete(keyList2.get(sel_bgID));
						if(!conflictPairs.containsKey(sel_bgID))
						{
							
							conflictPairs.put(sel_bgID, new LinkedList<Integer>());
						}
						conflictPairs.get(sel_bgID).add(j);
						

					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}

				}
				QueryIds.clear();
				for(Map.Entry<Integer,LinkedList<Integer>> entry : conflictPairs.entrySet()) 
				{
					int sel_bgID=entry.getKey();
					Iterator<Integer> iter = entry.getValue().iterator();
					int j=-1;
					double min_distance=Double.MAX_VALUE;
					while(iter.hasNext())
					{
						int curr1=iter.next();
						double curr_dist1=calculateDistance(keyList2.get(sel_bgID),keyList1.get(curr1));
						if(curr_dist1<min_distance)
						{
							min_distance=curr_dist1;
							j=curr1;
						}
					}
					if(min_distance<max_allow_dist)
					{
						
						try {
							KDTree2.delete(keyList2.get(sel_bgID));
						} catch (KeySizeException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						} catch (KeyMissingException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
						retIDs.get(0).add(j);
						retIDs.get(1).add(sel_bgID);
						for (int i = 0; i < numFeat; i++) {
							FeatVals_resample.get(i).add(FeatVals2.get(i).get(sel_bgID));
							FeatVals_resample0.get(i).add(FeatVals1.get(i).get(j));
						}
						//for other j , add to next round
						iter = entry.getValue().iterator();
						while(iter.hasNext())
						{
							int curr1=iter.next();
							if(curr1!=j)
							{
								QueryIds.add(curr1);
							}
						}
						
					}

					
				}
				if(FeatVals_resample.get(0).size()<3)
					break;
				//print out statistics compare re-sampling result
				ArrayList<EmpiricalDistribution> featDistLists_resample=new ArrayList<EmpiricalDistribution>();
				ArrayList<EmpiricalDistribution> featDistLists_resample0=new ArrayList<EmpiricalDistribution>();
				for (int i = 0; i < FeatVals1.size(); i++) {
					EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numbin);
					CGdistribution.load(ArrayUtils.toPrimitive(FeatVals_resample.get(i).toArray(new Double[0])));
					featDistLists_resample.add(CGdistribution);
					
					EmpiricalDistribution CGdistribution0=new EmpiricalDistribution(numbin);
					CGdistribution0.load(ArrayUtils.toPrimitive(FeatVals_resample0.get(i).toArray(new Double[0])));
					featDistLists_resample0.add(CGdistribution0);
				}

				System.err.println("mean/std: Sample1\tRe-sample2\t"+FeatVals_resample.get(0).size()+" Samples");
				for (int i = 0; i < numFeat; i++) {
					String out_s="Feature "+i+" : ";
					out_s+=featDistLists_resample0.get(i).getSampleStats().getMean()+"/"+featDistLists_resample0.get(i).getSampleStats().getStandardDeviation();
					out_s+="\t"+featDistLists_resample.get(i).getSampleStats().getMean()+"/"+featDistLists_resample.get(i).getSampleStats().getStandardDeviation();
					System.err.println(out_s);
				}
				
				if(QueryIds.size()==0)
					break;
			}

			return retIDs;
		}
	
	//this function directly take the joint feature to query, instead of independently generate each dimension from empirical
	public static ArrayList<ArrayList<Integer>> getBalanceSetIDs_jointDist(List<double[]> keyList1,List<double[]> keyList2 )
	{
		numFeat=keyList1.get(0).length; //resure the numfeat
		ArrayList<ArrayList<Integer>> retIDs=new ArrayList<ArrayList<Integer>>();
		for (int i = 0; i <2; i++) {
			retIDs.add(new ArrayList<Integer>());
		}
		
		ArrayList<ArrayList<Double>> FeatVals1=new ArrayList<ArrayList<Double>>(numFeat); //use to produce empirical distribution
		ArrayList<ArrayList<Double>> FeatVals2=new ArrayList<ArrayList<Double>>(numFeat);
		ArrayList<ArrayList<Double>> FeatVals_resample=new ArrayList<ArrayList<Double>>(numFeat);
		KDTree<Integer> KDTree1=new KDTree<Integer>(numFeat);
		KDTree<Integer> KDTree2=new KDTree<Integer>(numFeat);
		for (int i = 0; i < numFeat; i++) {
			FeatVals1.add(new ArrayList<Double>());
			FeatVals2.add(new ArrayList<Double>());
			FeatVals_resample.add(new ArrayList<Double>());
		}
		Random rand=new Random(0);
		for (int i = 0; i < keyList1.size(); i++) {
			double[] feat=keyList1.get(i);
			for (int j= 0; j < numFeat; j++) {
				 feat[j] += (rand.nextDouble()-0.5)*jitter;
				 FeatVals1.get(j).add(feat[j] );
			}
			try {
				keyList1.set(i, feat);
				if(keyList1.size()>keyList2.size())
				KDTree1.insert(feat,KDTree1.size());
			} catch (KeySizeException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (KeyDuplicateException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < keyList2.size(); i++) {
			double[] feat=keyList2.get(i);
			for (int j= 0; j < numFeat; j++) {
				 feat[j] += (rand.nextDouble()-0.5)*jitter;
				 FeatVals2.get(j).add(feat[j] );
			}
			try {
				keyList2.set(i, feat);
				if(keyList2.size()>keyList1.size())
				KDTree2.insert(feat,KDTree2.size());
			} catch (KeySizeException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (KeyDuplicateException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		if(KDTree2.size()>KDTree1.size())
		{
			ArrayList<EmpiricalDistribution> featDistLists=new ArrayList<EmpiricalDistribution>();
			for (int i = 0; i < FeatVals1.size(); i++) {
				EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numbin);
				CGdistribution.load(ArrayUtils.toPrimitive(FeatVals1.get(i).toArray(new Double[0])));
				featDistLists.add(CGdistribution);
			}
			int numQuerys = Math.min(maxSampleSize,  keyList1.size());
			for (int j = 0; j < numQuerys; j++) {
				double [] query=keyList1.get(j);

				int sel_bgID;
				try {
					sel_bgID = KDTree2.nearest(query);
					KDTree2.delete(keyList2.get(sel_bgID));
					retIDs.get(0).add(j);
					retIDs.get(1).add(sel_bgID);
					for (int i = 0; i < numFeat; i++) {
						FeatVals_resample.get(i).add(FeatVals2.get(i).get(sel_bgID));
					}
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

			}
			//print out statistics compare re-sampling result
			ArrayList<EmpiricalDistribution> featDistLists_resample=new ArrayList<EmpiricalDistribution>();
			for (int i = 0; i < FeatVals1.size(); i++) {
				EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numbin);
				CGdistribution.load(ArrayUtils.toPrimitive(FeatVals_resample.get(i).toArray(new Double[0])));
				featDistLists_resample.add(CGdistribution);
			}
			System.err.println("mean/std: Sample1\tRe-sample2");
			for (int i = 0; i < numFeat; i++) {
				String out_s="Feature "+i+" : ";
				out_s+=featDistLists.get(i).getSampleStats().getMean()+"/"+featDistLists.get(i).getSampleStats().getStandardDeviation();
				out_s+="\t"+featDistLists_resample.get(i).getSampleStats().getMean()+"/"+featDistLists_resample.get(i).getSampleStats().getStandardDeviation();
				System.err.println(out_s);
			}
			
		}
		else
		{
			
			ArrayList<EmpiricalDistribution> featDistLists=new ArrayList<EmpiricalDistribution>();
			for (int i = 0; i < FeatVals2.size(); i++) {
				EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numbin);
				CGdistribution.load(ArrayUtils.toPrimitive(FeatVals2.get(i).toArray(new Double[0])));
				featDistLists.add(CGdistribution);
			}
			int numQuerys = Math.min(maxSampleSize,  keyList2.size());
			for (int j = 0; j <numQuerys; j++) {
				double [] query=keyList2.get(j);
				int sel_bgID;
				try {
					sel_bgID = KDTree1.nearest(query);
					KDTree1.delete(keyList1.get(sel_bgID));
					retIDs.get(1).add(j);
					retIDs.get(0).add(sel_bgID);
					for (int i = 0; i < numFeat; i++) {
						FeatVals_resample.get(i).add(FeatVals1.get(i).get(sel_bgID));
					}
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

			}
			//print out statistics compare re-sampling result
			ArrayList<EmpiricalDistribution> featDistLists_resample=new ArrayList<EmpiricalDistribution>();
			for (int i = 0; i < FeatVals1.size(); i++) {
				EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numbin);
				CGdistribution.load(ArrayUtils.toPrimitive(FeatVals_resample.get(i).toArray(new Double[0])));
				featDistLists_resample.add(CGdistribution);
			}
			System.err.println("mean/std: Sample2\tRe-sample1");
			for (int i = 0; i < numFeat; i++) {
				String out_s="Feature "+i+" : ";
				out_s+=featDistLists.get(i).getSampleStats().getMean()+"/"+featDistLists.get(i).getSampleStats().getStandardDeviation();
				out_s+="\t"+featDistLists_resample.get(i).getSampleStats().getMean()+"/"+featDistLists_resample.get(i).getSampleStats().getStandardDeviation();
				System.err.println(out_s);
			}
		}
		return retIDs;
	}
	
	public static ArrayList<ArrayList<Integer>> getBalanceSetIDs(List<double[]> keyList1,List<double[]> keyList2 )
	{
		numFeat=keyList1.get(0).length; //resure the numfeat
		ArrayList<ArrayList<Integer>> retIDs=new ArrayList<ArrayList<Integer>>();
		for (int i = 0; i <2; i++) {
			retIDs.add(new ArrayList<Integer>());
		}
		
		ArrayList<ArrayList<Double>> FeatVals1=new ArrayList<ArrayList<Double>>(numFeat); //use to produce empirical distribution
		ArrayList<ArrayList<Double>> FeatVals2=new ArrayList<ArrayList<Double>>(numFeat);
		ArrayList<ArrayList<Double>> FeatVals_resample=new ArrayList<ArrayList<Double>>(numFeat);
		KDTree<Integer> KDTree1=new KDTree<Integer>(numFeat);
		KDTree<Integer> KDTree2=new KDTree<Integer>(numFeat);
		for (int i = 0; i < numFeat; i++) {
			FeatVals1.add(new ArrayList<Double>());
			FeatVals2.add(new ArrayList<Double>());
			FeatVals_resample.add(new ArrayList<Double>());
		}
		Random rand=new Random(0);
		for (int i = 0; i < keyList1.size(); i++) {
			double[] feat=keyList1.get(i);
			for (int j= 0; j < numFeat; j++) {
				 feat[j] += (rand.nextDouble()-0.5)*jitter;
				 FeatVals1.get(j).add(feat[j] );
			}
			try {
				keyList1.set(i, feat);
				if(keyList1.size()>keyList2.size())
				KDTree1.insert(feat,KDTree1.size());
			} catch (KeySizeException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (KeyDuplicateException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < keyList2.size(); i++) {
			double[] feat=keyList2.get(i);
			for (int j= 0; j < numFeat; j++) {
				 feat[j] += (rand.nextDouble()-0.5)*jitter;
				 FeatVals2.get(j).add(feat[j] );
			}
			try {
				keyList2.set(i, feat);
				if(keyList2.size()>keyList1.size())
				KDTree2.insert(feat,KDTree2.size());
			} catch (KeySizeException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (KeyDuplicateException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		if(KDTree2.size()>KDTree1.size())
		{
			ArrayList<EmpiricalDistribution> featDistLists=new ArrayList<EmpiricalDistribution>();
			for (int i = 0; i < FeatVals1.size(); i++) {
				EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numbin);
				CGdistribution.load(ArrayUtils.toPrimitive(FeatVals1.get(i).toArray(new Double[0])));
				featDistLists.add(CGdistribution);
			}
			for (int j = 0; j < keyList1.size(); j++) {
				double [] query=new double[FeatVals1.size()];
				for (int i = 0; i < FeatVals1.size(); i++) {
					query[i]=featDistLists.get(i).getNextValue();
				}
				int sel_bgID;
				try {
					sel_bgID = KDTree2.nearest(query);
					KDTree2.delete(keyList2.get(sel_bgID));
					retIDs.get(0).add(j);
					retIDs.get(1).add(sel_bgID);
					for (int i = 0; i < numFeat; i++) {
						FeatVals_resample.get(i).add(FeatVals2.get(i).get(sel_bgID));
					}
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

			}
			//print out statistics compare re-sampling result
			ArrayList<EmpiricalDistribution> featDistLists_resample=new ArrayList<EmpiricalDistribution>();
			for (int i = 0; i < FeatVals1.size(); i++) {
				EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numbin);
				CGdistribution.load(ArrayUtils.toPrimitive(FeatVals_resample.get(i).toArray(new Double[0])));
				featDistLists_resample.add(CGdistribution);
			}
			System.err.println("mean/std: Sample1\tRe-sample2");
			for (int i = 0; i < numFeat; i++) {
				String out_s="Feature "+i+" : ";
				out_s+=featDistLists.get(i).getSampleStats().getMean()+"/"+featDistLists.get(i).getSampleStats().getStandardDeviation();
				out_s+="\t"+featDistLists_resample.get(i).getSampleStats().getMean()+"/"+featDistLists_resample.get(i).getSampleStats().getStandardDeviation();
				System.err.println(out_s);
			}
			
		}
		else
		{
			
			ArrayList<EmpiricalDistribution> featDistLists=new ArrayList<EmpiricalDistribution>();
			for (int i = 0; i < FeatVals2.size(); i++) {
				EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numbin);
				CGdistribution.load(ArrayUtils.toPrimitive(FeatVals2.get(i).toArray(new Double[0])));
				featDistLists.add(CGdistribution);
			}
			for (int j = 0; j < keyList2.size(); j++) {
				double [] query=new double[FeatVals2.size()];
				for (int i = 0; i < FeatVals2.size(); i++) {
					query[i]=featDistLists.get(i).getNextValue();
				}
				int sel_bgID;
				try {
					sel_bgID = KDTree1.nearest(query);
					KDTree1.delete(keyList1.get(sel_bgID));
					retIDs.get(1).add(j);
					retIDs.get(0).add(sel_bgID);
					for (int i = 0; i < numFeat; i++) {
						FeatVals_resample.get(i).add(FeatVals1.get(i).get(sel_bgID));
					}
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

			}
			//print out statistics compare re-sampling result
			ArrayList<EmpiricalDistribution> featDistLists_resample=new ArrayList<EmpiricalDistribution>();
			for (int i = 0; i < FeatVals1.size(); i++) {
				EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numbin);
				CGdistribution.load(ArrayUtils.toPrimitive(FeatVals_resample.get(i).toArray(new Double[0])));
				featDistLists_resample.add(CGdistribution);
			}
			System.err.println("mean/std: Sample2\tRe-sample1");
			for (int i = 0; i < numFeat; i++) {
				String out_s="Feature "+i+" : ";
				out_s+=featDistLists.get(i).getSampleStats().getMean()+"/"+featDistLists.get(i).getSampleStats().getStandardDeviation();
				out_s+="\t"+featDistLists_resample.get(i).getSampleStats().getMean()+"/"+featDistLists_resample.get(i).getSampleStats().getStandardDeviation();
				System.err.println(out_s);
			}
		}

		
		return retIDs;
	}
	
	public static void PingPongDemo()
	{
		DemoFlag=true;
		NormalDistribution NormdistX1=new NormalDistribution(-1, 1);
		NormalDistribution NormdistX2=new NormalDistribution(0, 3);
		NormalDistribution NormdistY1=new NormalDistribution(-1, 2);
		NormalDistribution NormdistY2=new NormalDistribution(0, 3);
		int sample1Num=1000;
		int sample2Num=3000;
		ArrayList<double[]> sample1=new ArrayList<double[]>(sample1Num);
		ArrayList<double[]> sample2=new ArrayList<double[]>(sample2Num);
		for (int i = 0; i < sample1Num; i++) {
			double[] t =new double[2];
			t[0]=NormdistX1.sample();
			t[1]=NormdistY1.sample();
			sample1.add(t);
		}
		for (int i = 0; i < sample2Num; i++) {
			double[] t =new double[2];
			t[0]=NormdistX2.sample();
			t[1]=NormdistY2.sample();
			sample2.add(t);
		}
		
		getBalanceSetIDs_jointDistIter(sample1, sample2);
	}

	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
			// TODO Auto-generated method stub
//		if(true)
//		{
//		PingPongDemo();
//		return ;
//		}

			Options options = new Options();
			CommandLineParser parser = new GnuParser();
			ArrayList<double[]> keyList1=new ArrayList<double[]>();
			ArrayList<double[]> keyList2=new ArrayList<double[]>();
			ArrayList<String> lines1=new ArrayList<String>();
			ArrayList<String> lines2=new ArrayList<String>();
			CommandLine cmd;
			String inputfile1="";
			String inputfile2="";
			boolean jointFeatures=false;
			boolean pingpong=false;
			options.addOption("sample1", true, "tab separated file: ID\tfeature1...");
			options.addOption("sample2", true, "tab separated file: ID\tfeature1...");
			options.addOption("numFeat", true, "number of features you want to consider, default:2");
			options.addOption("EPS", true, "stop criteria for pingpong method, default:0.01");
			options.addOption("jitter", true, "default is 0.0001, change it to adjust the randomizing effect in the resampling");
			options.addOption("numbin", true, "default is 100, change it to adjust how precise to model the forground distribution");
			options.addOption("joint", false, "add this option, the program assume each feature is NOT independent, and need more number of background samples!");
			options.addOption("pingpong", false, "add this option, the program will iteractive subsample both samples to find the best match sets!");
			
			try {
				cmd = parser.parse( options, args);
				if(cmd.hasOption("sample1"))
				{
					inputfile1=cmd.getOptionValue("sample1");
				}
				else
				{
					HelpFormatter formatter = new HelpFormatter();
					formatter.printHelp( "BalanceSampler", options );
					return;
				}
				if(cmd.hasOption("sample2"))
				{
					inputfile2=cmd.getOptionValue("sample2");
				}
				else
				{
					HelpFormatter formatter = new HelpFormatter();
					formatter.printHelp( "BalanceSampler", options );
					return;
				}
				if(cmd.hasOption("numFeat"))
				{
					numFeat=Integer.valueOf(cmd.getOptionValue("numFeat"));
				}
				if(cmd.hasOption("jitter"))
				{
					jitter=Double.valueOf(cmd.getOptionValue("jitter"));
				}
				if(cmd.hasOption("EPS"))
				{
					EPS=Double.valueOf(cmd.getOptionValue("EPS"));
				}
				if(cmd.hasOption("numbin"))
				{
					numbin=Integer.valueOf(cmd.getOptionValue("numbin"));
				}

				if(cmd.hasOption("joint"))
					jointFeatures=true;
				if(cmd.hasOption("pingpong"))
					pingpong=true;
				
			} catch (ParseException e) {
				// TODO Auto-generated catch block
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp( "BalanceSampler", options );
				return;
			}
			

			Random rand=new Random(0);
			//read file 1
			BufferedReader br;
			try {
				br = new BufferedReader(new FileReader(inputfile1));
				String line;
				while ((line = br.readLine()) != null) {
				   // process the line.
					String[] comps = line.split("\t");
					double[] feat=new double[numFeat];
					for (int i = 1; i <= numFeat; i++) {
						double temp = Double.parseDouble(comps[i])+(rand.nextDouble()-0.5)*jitter;
						
						feat[i-1]=temp;
					}
					try {
						keyList1.add(feat);
						lines1.add(line);
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				br.close();
				
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			
			//read file 2
			try {
				br = new BufferedReader(new FileReader(inputfile2));
				String line;
				while ((line = br.readLine()) != null) {
				   // process the line.
					String[] comps = line.split("\t");
					double[] feat=new double[numFeat];
					for (int i = 1; i <= numFeat; i++) {
						double temp = Double.parseDouble(comps[i])+(rand.nextDouble()-0.5)*jitter;
						feat[i-1]=temp;
					}
					try {
						keyList2.add(feat);
						lines2.add(line);
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				br.close();
				
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			numbin=Math.min(numbin,Math.min(keyList2.size(), keyList1.size())/30);
			ArrayList<ArrayList<Integer>> balanceIDs = null;
			if(pingpong)
				balanceIDs=getBalanceSetIDs_jointDistIter(keyList1, keyList2);
			else if(jointFeatures)
				balanceIDs=getBalanceSetIDs_jointDist_bound(keyList1, keyList2,EPS);
			else
				balanceIDs=getBalanceSetIDs(keyList1, keyList2);
	
			
			
			//print the samplig sequence
			for (int i = 0; i < balanceIDs.get(0).size(); i++) {
				System.out.println(lines1.get(balanceIDs.get(0).get(i)));
			}
			
			for (int i = 0; i < balanceIDs.get(1).size(); i++) {
				System.out.println(lines2.get(balanceIDs.get(1).get(i)));
			}
			

	}
	
	public static void main_old(String[] args) {
		// TODO Auto-generated method stub
			// TODO Auto-generated method stub
			Options options = new Options();
			CommandLineParser parser = new GnuParser();
			ArrayList<double[]> keyList1=new ArrayList<double[]>();
			ArrayList<double[]> keyList2=new ArrayList<double[]>();
			ArrayList<String> lines1=new ArrayList<String>();
			ArrayList<String> lines2=new ArrayList<String>();
			CommandLine cmd;
			String inputfile1="";
			String inputfile2="";
			
			options.addOption("sample1", true, "tab separated file: ID\tfeature1...");
			options.addOption("sample2", true, "tab separated file: ID\tfeature1...");
			options.addOption("numFeat", true, "number of features you want to consider, default:2");
			options.addOption("jitter", true, "default is 0.0001, change it to adjust the randomizing effect in the resampling");
			options.addOption("numbin", true, "default is 100, change it to adjust how precise to model the forground distribution");
			
			
			
			try {
				cmd = parser.parse( options, args);
				if(cmd.hasOption("sample1"))
				{
					inputfile1=cmd.getOptionValue("sample1");
				}
				else
				{
					HelpFormatter formatter = new HelpFormatter();
					formatter.printHelp( "BalanceSampler", options );
					return;
				}
				if(cmd.hasOption("sample2"))
				{
					inputfile2=cmd.getOptionValue("sample2");
				}
				else
				{
					HelpFormatter formatter = new HelpFormatter();
					formatter.printHelp( "BalanceSampler", options );
					return;
				}
				if(cmd.hasOption("numFeat"))
				{
					numFeat=Integer.valueOf(cmd.getOptionValue("numFeat"));
				}
				if(cmd.hasOption("jitter"))
				{
					jitter=Double.valueOf(cmd.getOptionValue("jitter"));
				}
				if(cmd.hasOption("numbin"))
				{
					numbin=Integer.valueOf(cmd.getOptionValue("numbin"));
				}

				
			} catch (ParseException e) {
				// TODO Auto-generated catch block
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp( "BalanceSampler", options );
				return;
			}
			
			ArrayList<ArrayList<Double>> FeatVals1=new ArrayList<ArrayList<Double>>(numFeat); //use to produce empirical distribution
			ArrayList<ArrayList<Double>> FeatVals2=new ArrayList<ArrayList<Double>>(numFeat);
			ArrayList<ArrayList<Double>> FeatVals_resample=new ArrayList<ArrayList<Double>>(numFeat);
			KDTree<Integer> KDTree1=new KDTree<Integer>(numFeat);
			KDTree<Integer> KDTree2=new KDTree<Integer>(numFeat);
			for (int i = 0; i < numFeat; i++) {
				FeatVals1.add(new ArrayList<Double>());
				FeatVals2.add(new ArrayList<Double>());
				FeatVals_resample.add(new ArrayList<Double>());
			}
			Random rand=new Random(0);
			//read file 1
			BufferedReader br;
			try {
				br = new BufferedReader(new FileReader(inputfile1));
				String line;
				while ((line = br.readLine()) != null) {
				   // process the line.
					String[] comps = line.split("\t");
					double[] feat=new double[numFeat];
					for (int i = 1; i <= numFeat; i++) {
						double temp = Double.parseDouble(comps[i])+(rand.nextDouble()-0.5)*jitter;
						FeatVals1.get(i-1).add(temp);
						feat[i-1]=temp;
					}
					try {

						KDTree1.insert(feat, keyList1.size());
						keyList1.add(feat);
						lines1.add(line);
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				br.close();
				
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			
			//read file 2
			try {
				br = new BufferedReader(new FileReader(inputfile2));
				String line;
				while ((line = br.readLine()) != null) {
				   // process the line.
					String[] comps = line.split("\t");
					double[] feat=new double[numFeat];
					for (int i = 1; i <= numFeat; i++) {
						double temp = Double.parseDouble(comps[i])+(rand.nextDouble()-0.5)*jitter;
						FeatVals2.get(i-1).add(temp);
						feat[i-1]=temp;
					}
					try {

						KDTree2.insert(feat, keyList2.size());
						keyList2.add(feat);
						lines2.add(line);
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				br.close();
				
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			ArrayList<String> balance_samples=new ArrayList<String>();
			 numbin=Math.min(numbin,Math.min(KDTree2.size(), KDTree1.size())/30);
			if(KDTree2.size()>KDTree1.size())
			{
				ArrayList<EmpiricalDistribution> featDistLists=new ArrayList<EmpiricalDistribution>();
				for (int i = 0; i < FeatVals1.size(); i++) {
					EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numbin);
					CGdistribution.load(ArrayUtils.toPrimitive(FeatVals1.get(i).toArray(new Double[0])));
					featDistLists.add(CGdistribution);
				}
				for (int j = 0; j < KDTree1.size(); j++) {
					double [] query=new double[FeatVals1.size()];
					for (int i = 0; i < FeatVals1.size(); i++) {
						query[i]=featDistLists.get(i).getNextValue();
					}
					int sel_bgID;
					try {
						sel_bgID = KDTree2.nearest(query);
						KDTree2.delete(keyList2.get(sel_bgID));
						balance_samples.add(lines2.get(sel_bgID)); //add bg sampl
						balance_samples.add(lines1.get(j)); //add positive sampl
						for (int i = 0; i < numFeat; i++) {
							FeatVals_resample.get(i).add(FeatVals2.get(i).get(sel_bgID));
						}
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}

				}
				//print out statistics compare re-sampling result
				ArrayList<EmpiricalDistribution> featDistLists_resample=new ArrayList<EmpiricalDistribution>();
				for (int i = 0; i < FeatVals1.size(); i++) {
					EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numbin);
					CGdistribution.load(ArrayUtils.toPrimitive(FeatVals_resample.get(i).toArray(new Double[0])));
					featDistLists_resample.add(CGdistribution);
				}
				System.err.println("mean/std: Sample1\tRe-sample2");
				for (int i = 0; i < numFeat; i++) {
					String out_s="Feature "+i+" : ";
					out_s+=featDistLists.get(i).getSampleStats().getMean()+"/"+featDistLists.get(i).getSampleStats().getStandardDeviation();
					out_s+="\t"+featDistLists_resample.get(i).getSampleStats().getMean()+"/"+featDistLists_resample.get(i).getSampleStats().getStandardDeviation();
					System.err.println(out_s);
				}
				
			}
			else
			{
				
				ArrayList<EmpiricalDistribution> featDistLists=new ArrayList<EmpiricalDistribution>();
				for (int i = 0; i < FeatVals2.size(); i++) {
					EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numbin);
					CGdistribution.load(ArrayUtils.toPrimitive(FeatVals2.get(i).toArray(new Double[0])));
					featDistLists.add(CGdistribution);
				}
				for (int j = 0; j < KDTree2.size(); j++) {
					double [] query=new double[FeatVals2.size()];
					for (int i = 0; i < FeatVals2.size(); i++) {
						query[i]=featDistLists.get(i).getNextValue();
					}
					int sel_bgID;
					try {
						sel_bgID = KDTree1.nearest(query);
						KDTree1.delete(keyList1.get(sel_bgID));
						balance_samples.add(lines1.get(sel_bgID)); //add bg sampl
						balance_samples.add(lines2.get(j)); //add positive sampl
						for (int i = 0; i < numFeat; i++) {
							FeatVals_resample.get(i).add(FeatVals1.get(i).get(sel_bgID));
						}
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}

				}
				//print out statistics compare re-sampling result
				ArrayList<EmpiricalDistribution> featDistLists_resample=new ArrayList<EmpiricalDistribution>();
				for (int i = 0; i < FeatVals1.size(); i++) {
					EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numbin);
					CGdistribution.load(ArrayUtils.toPrimitive(FeatVals_resample.get(i).toArray(new Double[0])));
					featDistLists_resample.add(CGdistribution);
				}
				System.err.println("mean/std: Sample2\tRe-sample1");
				for (int i = 0; i < numFeat; i++) {
					String out_s="Feature "+i+" : ";
					out_s+=featDistLists.get(i).getSampleStats().getMean()+"/"+featDistLists.get(i).getSampleStats().getStandardDeviation();
					out_s+="\t"+featDistLists_resample.get(i).getSampleStats().getMean()+"/"+featDistLists_resample.get(i).getSampleStats().getStandardDeviation();
					System.err.println(out_s);
				}
			}
	
			
			
			//print the samplig sequence
			for (int i = 0; i < balance_samples.size(); i++) {
				System.out.println(balance_samples.get(i));
			}
			
	

	}

}
