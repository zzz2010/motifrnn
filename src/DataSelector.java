import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.math3.random.EmpiricalDistribution;
import org.ejml.simple.SimpleMatrix;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.LabeledWord;
import edu.stanford.nlp.sentiment.RNNOptions;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.ArrayUtils;
import edu.wlu.cs.levy.CG.KDTree;
import edu.wlu.cs.levy.CG.KeyDuplicateException;
import edu.wlu.cs.levy.CG.KeyMissingException;
import edu.wlu.cs.levy.CG.KeySizeException;

//generate balance dataset, with similar length and CG content
// convert weka format
public class DataSelector {
	
	static HashMap<String, Float> kmerPrior=null;
	public static Instances Tree2Weka(String filepath, int class_index)
	{
		
		List<Tree> trainSample = BalanceSamples(filepath,class_index);
		
		return Tree2Weka_(trainSample,class_index);
	}
	
	//for flatformat input
	public static List<String> BalanceSamples(String filepath, String class_str)
	{
		System.err.println("BalanceSamples......");
		System.out.println("CGCorrectionOption: "+Utils.CGcorrection);
		System.out.println("LengthMappabilityCorrectionOption: "+Utils.lengthMappabilityCorrection);
	
		ArrayList<String> Positie_samples=new ArrayList<String>();
		ArrayList<String> BG_samples=new ArrayList<String>();
		ArrayList<String> balance_samples=new ArrayList<String>();

		ArrayList<double[]> point_KDTree1=new ArrayList<double[]>();
		ArrayList<double[]> point_KDTree2=new ArrayList<double[]>();
		
		int toolongSamples=0;
		int notMappableSamples=0;
		int numBGsamples=0;
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(filepath));
			String line;
			while ((line = br.readLine()) != null) {
			   // process the line.
				String[] comps = line.split("\t");
				double target=0;
				boolean morethan1Class=false;
				if(comps.length>4&&comps[4].contains(class_str))
				{
					target=1;
					if(comps[4].length()>class_str.length())
						morethan1Class=true;
				}
				double cg=0;
				double len=0;
				double[] cglenmap=new double[4];
				boolean tooLong=false;
				boolean tooGCbias=false;
				boolean notMappable=false;
				for(String m_str:comps[3].split(";"))
				{
					if(m_str.startsWith("CG_content"))
					{
						cg=Double.valueOf(m_str.split(":")[2]);
						if(!Utils.studyGC_CpG_Repeat&&Utils.CGcorrection)
							cglenmap[0]=cg;
//						if(Math.abs(cg-0.5)>0.1)
//							tooGCbias=true;
					}
					else if(m_str.startsWith("Length")&&Utils.lengthMappabilityCorrection)
					{
						double tmp=Double.valueOf(m_str.split(":")[2]);
						len=tmp*10;
						if(tmp>Utils.region_length_threshold) //i.e., filter too large regions
							tooLong=true;
						cglenmap[1]=len;
					}
					else if(m_str.startsWith("Mappability")&&Utils.lengthMappabilityCorrection)
					{
						double tmp=Double.valueOf(m_str.split(":")[2]);
						if(tmp<Utils.region_mappability_threshold)
							notMappable=true;
						cglenmap[2]=tmp;
					}
					else if(m_str.startsWith("CpG"))
					{
						Double cpg = Double.valueOf(m_str.split(":")[2]);
						if(!Utils.studyGC_CpG_Repeat&&Utils.CGcorrection)
							cglenmap[3]+=cpg;
					}
					
				}


				
				if(tooLong)
				{
					toolongSamples+=1;
					continue;
				}
				if(notMappable)
				{
					notMappableSamples+=1;
					continue;
				}
				if(tooGCbias)
				{
					continue;
				}
				if(target>=1)
				{
					Positie_samples.add(line);
					point_KDTree1.add(cglenmap);
					
				}
				else if(!Utils.useNoModificationRegionAsBG||!morethan1Class) //must no other histone signal there
				{
					numBGsamples+=1;
					point_KDTree2.add(cglenmap);
					BG_samples.add(line);
				}
			}
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		if(!Utils.CGcorrection&&!Utils.lengthMappabilityCorrection) //no correction
		{
			balance_samples.addAll(Positie_samples);
			balance_samples.addAll(BG_samples);
			return balance_samples;
		}
		
		BalanceSampler.maxSampleSize=Utils.max_train_sample_size/2;
		ArrayList<ArrayList<Integer>> balanceIDs =BalanceSampler.getBalanceSetIDs_jointDistIter(point_KDTree1, point_KDTree2);
		//return the samplig sequence
		for (int i = 0; i < balanceIDs.get(0).size(); i++) {
			balance_samples.add(Positie_samples.get(balanceIDs.get(0).get(i)));
		}
		
		for (int i = 0; i < balanceIDs.get(1).size(); i++) {
			balance_samples.add(BG_samples.get(balanceIDs.get(1).get(i)));
		}
				return balance_samples;
	}
	
	public static List<String> BalanceSamples_old(String filepath, String class_str)
	{
		System.err.println("BalanceSamples......");
		 int numBin=10;
		EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numBin);
		EmpiricalDistribution Lengthdistribution=new EmpiricalDistribution(numBin);
		ArrayList<Double> CGvalues=new ArrayList<Double>();
		ArrayList<Double> Lenvalues=new ArrayList<Double>();
		ArrayList<String> Positie_samples=new ArrayList<String>();
		ArrayList<String> BG_samples=new ArrayList<String>();
		ArrayList<String> balance_samples=new ArrayList<String>();
		ArrayList<Double> bgCGvalues=new ArrayList<Double>();
		ArrayList<Double> bgLenvalues=new ArrayList<Double>();
		ArrayList<double[]> point_KDTree=new ArrayList<double[]>();
		
		KDTree<Integer> bgTree=new KDTree<Integer>(2);
		Random rand=new Random(0);
		int toolongSamples=0;
		int numBGsamples=0;
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(filepath));
			String line;
			while ((line = br.readLine()) != null) {
			   // process the line.
				String[] comps = line.split("\t");
				double target=0;
				boolean morethan1Class=false;
				if(comps.length>4&&comps[4].contains(class_str))
				{
					target=1;
					if(comps[4].length()>class_str.length())
						morethan1Class=true;
				}
				double cg=0;
				double len=0;
				double[] cglen=new double[2];
				boolean tooLong=false;
				for(String m_str:comps[3].split(";"))
				{
					if(m_str.contains("CG_content"))
					{
						cg=Double.valueOf(m_str.split(":")[2])+(rand.nextDouble()-0.5)/10000;
					}
					else if(m_str.contains("Length"))
					{
						double tmp=Double.valueOf(m_str.split(":")[2]);
						len=tmp*100+(rand.nextDouble()-0.5)/10000;
						if(tmp>Utils.region_length_threshold) //i.e., filter too large regions
							tooLong=true;
					}
				}

				cglen[0]=cg;
				cglen[1]=len;
				if(tooLong)
				{toolongSamples+=1;
					continue;
				}
				if(target>=1)
				{
					Positie_samples.add(line);
					CGvalues.add(cg);
					Lenvalues.add(len);
					
				}
				else if(!Utils.useNoModificationRegionAsBG||!morethan1Class) //must no other histone signal there
				{
					numBGsamples+=1;
					bgCGvalues.add(cg);
					bgLenvalues.add(len);
					
					try {
						//record cglen vec, so that later can easy delete from KDtree
						bgTree.insert(cglen, BG_samples.size());
						point_KDTree.add(cglen);
						BG_samples.add(line);
					} catch (KeySizeException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} catch (KeyDuplicateException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
			
//		System.err.println(toolongSamples);
		CGdistribution.load(ArrayUtils.toPrimitive(CGvalues.toArray(new Double[0])));
		Lengthdistribution.load(ArrayUtils.toPrimitive(Lenvalues.toArray(new Double[0])));
		
		//shuffle positive samples
		Collections.shuffle(Positie_samples);
		
		int num_posSamples=Math.min(Utils.max_train_sample_size/2,Positie_samples.size());
		double positiveSampleRate=(bgTree.size()*1.0)/num_posSamples;  //the reason for 0.5 here, to avoid oversampling from background
		for (int i = 0; i < num_posSamples; i++) {
			if(rand.nextDouble()>positiveSampleRate)
				continue;
			double[] cglen=new double[2];
			cglen[0]=CGdistribution.getNextValue();
			cglen[1]=Lengthdistribution.getNextValue();
			try {
				if(bgTree.size()==0)
					break;
				int sel_bgID = bgTree.nearest(cglen);
				balance_samples.add(BG_samples.get(sel_bgID)); //add bg sampl
				balance_samples.add(Positie_samples.get(i)); //add positive sampl
				//delete from KDTree
				double [] cgvec=point_KDTree.get(sel_bgID); 

				try {
					bgTree.delete(cgvec);
				} catch (KeyMissingException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				
			} catch (KeySizeException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
				return balance_samples;
	}
	
	public static List<Tree> BalanceSamples(String filepath, int class_index)
	{
		System.err.println("BalanceSamples......");
		 int numBin=10;
		EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numBin);
		EmpiricalDistribution Lengthdistribution=new EmpiricalDistribution(numBin);
		ArrayList<Double> CGvalues=new ArrayList<Double>();
		ArrayList<Double> Lenvalues=new ArrayList<Double>();
		ArrayList<String> Positie_samples=new ArrayList<String>();
		ArrayList<String> BG_samples=new ArrayList<String>();
		ArrayList<String> balance_samples=new ArrayList<String>();
		ArrayList<Double> bgCGvalues=new ArrayList<Double>();
		ArrayList<Double> bgLenvalues=new ArrayList<Double>();
		ArrayList<double[]> point_KDTree=new ArrayList<double[]>();
		
		KDTree<Integer> bgTree=new KDTree<Integer>(2);
		Random rand=new Random(0);
		int toolongSamples=0;
		int numBGsamples=0;
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(filepath));
			String line;
			while ((line = br.readLine()) != null) {
			   // process the line.
				String[] comps = line.split(" ",2);
				String[] target_strs=comps[0].substring(1).split(",");
				double target=Double.valueOf(target_strs[class_index]);
				double cg=0;
				double len=0;
				double[] cglen=new double[2];
				boolean tooLong=false;
				int pos1=line.indexOf(" CG_content");
				int pos2=line.lastIndexOf('(', pos1);
				cg=Double.valueOf(line.substring(pos2+1, pos1))+(rand.nextDouble()-0.5)/10000;
				 pos1=line.indexOf(" Length_");
				 pos2=line.lastIndexOf('(', pos1);
				double tmp=Double.valueOf(line.substring(pos2+1, pos1));
				len=tmp*100+(rand.nextDouble()-0.5)/10000;
				cglen[0]=cg;
				cglen[1]=len;
				if(tmp>Utils.region_length_threshold) //i.e., filter too large regions
					tooLong=true;
				if(tooLong)
				{toolongSamples+=1;
					continue;
				}
				if(target>=1)
				{
					Positie_samples.add(line);
					CGvalues.add(cg);
					Lenvalues.add(len);
					
				}
				else if(!Utils.useNoModificationRegionAsBG||!comps[0].contains("1")) //must no other histone signal there
				{
					numBGsamples+=1;
					bgCGvalues.add(cg);
					bgLenvalues.add(len);
					
					try {
						//record cglen vec, so that later can easy delete from KDtree
						bgTree.insert(cglen, BG_samples.size());
						point_KDTree.add(cglen);
						BG_samples.add(line);
					} catch (KeySizeException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} catch (KeyDuplicateException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
			
//		System.err.println(toolongSamples);
		CGdistribution.load(ArrayUtils.toPrimitive(CGvalues.toArray(new Double[0])));
		Lengthdistribution.load(ArrayUtils.toPrimitive(Lenvalues.toArray(new Double[0])));
		
		//shuffle positive samples
		Collections.shuffle(Positie_samples);
		
		int num_posSamples=Math.min(Utils.max_train_sample_size/2,Positie_samples.size());
		double positiveSampleRate=(bgTree.size()*1.0)/num_posSamples;  //the reason for 0.5 here, to avoid oversampling from background
		for (int i = 0; i < num_posSamples; i++) {
			if(rand.nextDouble()>positiveSampleRate)
				continue;
			double[] cglen=new double[2];
			cglen[0]=CGdistribution.getNextValue();
			cglen[1]=Lengthdistribution.getNextValue();
			try {
				if(bgTree.size()==0)
					break;
				int sel_bgID = bgTree.nearest(cglen);
				balance_samples.add(BG_samples.get(sel_bgID)); //add bg sampl
				balance_samples.add(Positie_samples.get(i)); //add positive sampl
				//delete from KDTree
				double [] cgvec=point_KDTree.get(sel_bgID); 

				try {
					bgTree.delete(cgvec);
				} catch (KeyMissingException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				
			} catch (KeySizeException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		

		//write output temp file
		String tempfile="tmpfile."+class_index;
		PrintWriter writer;
		try {
			writer = new  PrintWriter(new File(tempfile));
			for (int i = 0; i < balance_samples.size(); i++) {
				writer.println(balance_samples.get(i));
			}
			writer.close();
			List<Tree> trainSample=Utils.readTreesWithGoldLabels(tempfile);
			Files.delete(Paths.get(tempfile));
			return trainSample;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

				return null;
	}
	public static Instances FlatString2Weka(String filepath, String class_str)
	{
		List<String> trainSample = BalanceSamples(filepath,class_str);
		
		return FlatString2Weka_(trainSample,class_str);
	}
	
	
	
	static Instances FlatString2Weka_knownFeatureMap(List<String> trainSample, String class_str,HashMap<String, Integer> motif_Id)
	{
		System.gc() ;

		List<HashMap<String, Double>> flattenSamples=new ArrayList<HashMap<String, Double>>(trainSample.size());
		
		ArrayList<Double> targets=new ArrayList<Double>(trainSample.size());
		int mid=0;
		for (int i = 0; i < trainSample.size(); i++) {
			String sample_str = trainSample.get(i);
			HashMap<String, Double> sample=new HashMap<String, Double>();
			String[] comps = sample_str.split("\t");
			for(String motif_str :comps[3].split(";"))
			{
				String[] toks=motif_str.split(":");
				String motif_name=toks[0];
				double score=Double.parseDouble(toks[toks.length-1]);
//				 boolean cofounder=(motif_name.contains("content")||motif_name.contains("CpG")||motif_name.contains("Length")||motif_name.contains("Mapp"));
//				 if(!cofounder)
//					 continue;
				 
				if(Utils.excludeRepeat&&(motif_name.startsWith("Repeat_")||(motif_name.startsWith("CpGisland"))))
				{
					continue;
				}

				if(motif_name.startsWith("Length"))
					motif_name="Length";
				if(!motif_Id.containsKey(motif_name))
				{
					continue;
						
				}
				if(sample.containsKey(motif_name))
				{
					if(Utils.sumScore)
						sample.put(motif_name,sample.get(motif_name)+score);
					else
					sample.put(motif_name, Math.max(sample.get(motif_name),score));
					
				}
				else
				{
					sample.put(motif_name,score);
				}
			}
			
			double t=0;
			if(comps.length>4&&comps[4].contains(class_str))
				t=1;
			targets.add(t);
			
			flattenSamples.add(sample);
		}
		
		int featureNum=motif_Id.size();
		System.err.println("number of features : "+featureNum);

		FastVector attrList=new FastVector(featureNum+1);
		for (int i = 0; i < featureNum; i++) {
			attrList.addElement(new Attribute(""));
		}
		Iterator<Entry<String, Integer>> iter = motif_Id.entrySet().iterator();
		while(iter.hasNext())
		{
			Entry<String, Integer> name_id = iter.next();
			if(name_id.getKey().contains("target_value"))
				continue;
			Attribute temp=new Attribute(name_id.getKey());
//			if(name_id.getKey().contentEquals("MA04061TEC1"))
			attrList.setElementAt(temp, name_id.getValue());
//			attrList.addElement(temp);
		}
		
		FastVector Tvalues = new FastVector();
		Tvalues.addElement("negative");
		Tvalues.addElement("positive");
		Attribute label = new Attribute("target_value",Tvalues);
		
		attrList.addElement(label);
		int instanceNum=  flattenSamples.size();
		Instances weka_traindata=new Instances(class_str, attrList,instanceNum);
		for (int i = 0; i < instanceNum; i++) {
			 double[] values = new double[featureNum+1] ;
			 HashMap<String, Double> sample = flattenSamples.get(i);
			 double sumSQ=0;
			 for(String motifname:sample.keySet())
			 {
				 double tmp=sample.get(motifname);
				 values[motif_Id.get(motifname)]=tmp;
				 if(Utils.checkCoFounder(motifname))
					 continue;
				 else
					 sumSQ+=tmp*tmp;
			 } 
			 sumSQ=Math.sqrt(sumSQ);
			 if(!Utils.SQnormalization)
				 sumSQ=1;
			 for(String motifname:sample.keySet())
			 {
				 if(Utils.checkCoFounder(motifname))
				 {
					 continue; //skip the normalization for the cofounder
				 }
				 else
					 values[motif_Id.get(motifname)]/=sumSQ;
			 }
			
			 double labelval=0;
				 labelval=targets.get(i);
			if(labelval>=1)
				labelval=1;
			else
				labelval=0;

			 values[featureNum]=labelval;
			 Instance instance = new Instance(1, values);
			 weka_traindata.add(instance);
			
		}

		weka_traindata.setClassIndex(featureNum);
		
		System.err.println("prodcue"+weka_traindata.numInstances()+" training samples, class "+class_str);
		return weka_traindata;
	}
	
	static Instances FlatString2Weka_(List<String> trainSample, String class_str)
	{
		System.gc() ;

		List<HashMap<String, Double>> flattenSamples=new ArrayList<HashMap<String, Double>>(trainSample.size());
		HashMap<String, Integer> motif_Id=new HashMap<String, Integer>();
		ArrayList<Double> targets=new ArrayList<Double>(trainSample.size());
		int mid=0;
		for (int i = 0; i < trainSample.size(); i++) {
			String sample_str = trainSample.get(i);
			HashMap<String, Double> sample=new HashMap<String, Double>();
			String[] comps = sample_str.split("\t");
			for(String motif_str :comps[3].split(";"))
			{
				String[] toks=motif_str.split(":");
				String motif_name=toks[0];
				double score=Double.parseDouble(toks[toks.length-1]);
//				 boolean cofounder=(motif_name.contains("content")||motif_name.contains("CpG")||motif_name.contains("Length")||motif_name.contains("Mapp"));
//				 if(!cofounder)
//					 continue;
				 
				if(Utils.excludeRepeat&&(motif_name.startsWith("Repeat_")||(motif_name.startsWith("CpGisland"))))
				{
					continue;
				}

				if(motif_name.startsWith("Length"))
					motif_name="Length";
				if(!motif_Id.containsKey(motif_name))
				{
					
					motif_Id.put(motif_name, mid);
					mid+=1;
					
						
				}
				if(sample.containsKey(motif_name))
				{
					if(Utils.sumScore)
						sample.put(motif_name,sample.get(motif_name)+score);
					else
					sample.put(motif_name, Math.max(sample.get(motif_name),score));
					
				}
				else
				{
					sample.put(motif_name,score);
				}
			}
			
			double t=0;
			if(comps.length>4&&comps[4].contains(class_str))
				t=1;
			targets.add(t);
			
			flattenSamples.add(sample);
		}
		
		int featureNum=motif_Id.size();
		System.err.println("number of features : "+featureNum);

		FastVector attrList=new FastVector(featureNum+1);
		for (int i = 0; i < featureNum; i++) {
			attrList.addElement(new Attribute(""));
		}
		Iterator<Entry<String, Integer>> iter = motif_Id.entrySet().iterator();
		while(iter.hasNext())
		{
			Entry<String, Integer> name_id = iter.next();
			Attribute temp=new Attribute(name_id.getKey());
//			if(name_id.getKey().contentEquals("MA04061TEC1"))
			attrList.setElementAt(temp, name_id.getValue());
//			attrList.addElement(temp);
		}

//		System.out.println("Mappability id: "+motif_Id.get("Mappability"));
//		for (int i = 0; i < attrList.size(); i++) {
//			if(attrList.elementAt(i).toString().contains("Mappability"))
//			{
//				System.out.println("Mappability attr id: "+i);
//				break;
//			}
//		}
		
		
		
		FastVector Tvalues = new FastVector();
		Tvalues.addElement("negative");
		Tvalues.addElement("positive");
		Attribute label = new Attribute("target_value",Tvalues);
		
		attrList.addElement(label);
		int instanceNum=  flattenSamples.size();
		Instances weka_traindata=new Instances(class_str, attrList,instanceNum);
		for (int i = 0; i < instanceNum; i++) {
			 double[] values = new double[featureNum+1] ;
			 HashMap<String, Double> sample = flattenSamples.get(i);
			 double sumSQ=0;
			 for(String motifname:sample.keySet())
			 {
				 double tmp=sample.get(motifname);
				 values[motif_Id.get(motifname)]=tmp;
				 if(Utils.checkCoFounder(motifname))
					 continue;
				 else
					 sumSQ+=tmp*tmp;
			 } 
			 sumSQ=Math.sqrt(sumSQ);
			 if(!Utils.SQnormalization)
				 sumSQ=1;
			 for(String motifname:sample.keySet())
			 {
				 if(Utils.checkCoFounder(motifname))
				 {
					 continue; //skip the normalization for the cofounder
				 }
				 else
					 values[motif_Id.get(motifname)]/=sumSQ;
			 }
			
			 double labelval=0;
				 labelval=targets.get(i);
			if(labelval>=1)
				labelval=1;
			else
				labelval=0;

			 values[featureNum]=labelval;
			 Instance instance = new Instance(1, values);
			 weka_traindata.add(instance);
			
		}

		weka_traindata.setClassIndex(featureNum);
		
		System.err.println("prodcue"+weka_traindata.numInstances()+" training samples, class "+class_str);
		return weka_traindata;
	}
	private static Instances Tree2Weka_(List<Tree> trainSample, int class_index)
	{
		System.gc() ;

		List<HashMap<String, Double>> flattenSamples=new ArrayList<HashMap<String, Double>>(trainSample.size());
		HashMap<String, Integer> motif_Id=new HashMap<String, Integer>();
		ArrayList<Double> targets=new ArrayList<Double>(trainSample.size());
		int mid=0;
		for (int i = 0; i < trainSample.size(); i++) {
			Tree queryTree = trainSample.get(i);
			HashMap<String, Double> sample=new HashMap<String, Double>();
			targets.add(MotifRNNCoreAnnotations.getGoldClass(queryTree).get(class_index));
			
			for (LabeledWord leaf : queryTree.labeledYield()) {
				String[] comps = leaf.toString().split("/");
				String motif_name=comps[0];
				if(Utils.excludeRepeat&&motif_name.startsWith("Repeat_"))
				{
					continue;
				}
				if(!motif_Id.containsKey(motif_name))
				{
					
					motif_Id.put(motif_name, mid);
					mid+=1;
				}
				double score=Double.parseDouble(comps[comps.length-1]);
				if(sample.containsKey(motif_name))
				{
					if(Utils.sumScore)
						sample.put(motif_name,sample.get(motif_name)+score);
					else
						sample.put(motif_name, Math.max(sample.get(motif_name),score));
//					
				}
				else
				{
					sample.put(motif_name,score);
				}
			}
			flattenSamples.add(sample);
		}
		
		int featureNum=motif_Id.size();
		System.err.println("number of features : "+featureNum);

		FastVector attrList=new FastVector(featureNum+1);
		for (int i = 0; i < featureNum; i++) {
			attrList.addElement(new Attribute(""));
		}
		Iterator<Entry<String, Integer>> iter = motif_Id.entrySet().iterator();
		while(iter.hasNext())
		{
			Entry<String, Integer> name_id = iter.next();
			Attribute temp=new Attribute(name_id.getKey());
//			if(name_id.getKey().contentEquals("MA04061TEC1"))
			attrList.setElementAt(temp, name_id.getValue());
//			attrList.addElement(temp);
		}

		FastVector Tvalues = new FastVector();
		Tvalues.addElement("negative");
		Tvalues.addElement("positive");
		Attribute label = new Attribute("target_value",Tvalues);
		
		attrList.addElement(label);
		int instanceNum=  flattenSamples.size();
		Instances weka_traindata=new Instances(String.valueOf(class_index), attrList,instanceNum);
		for (int i = 0; i < instanceNum; i++) {
			 double[] values = new double[featureNum+1] ;
			 HashMap<String, Double> sample = flattenSamples.get(i);
			 double sumSQ=0;
			 for(String motifname:sample.keySet())
			 {
				 double tmp=sample.get(motifname);
				 values[motif_Id.get(motifname)]=tmp;
				 sumSQ+=tmp*tmp;
			 } 
			 sumSQ=Math.sqrt(sumSQ);
			 if(!Utils.SQnormalization)
				 sumSQ=1;
			 for(String motifname:sample.keySet())
			 {
				 values[motif_Id.get(motifname)]/=sumSQ;
			 }
			
			 double labelval=0;
				 labelval=targets.get(i);
			if(labelval>=1)
				labelval=1;
			else
				labelval=0;

			 values[featureNum]=labelval;
			 Instance instance = new Instance(1, values);
			 weka_traindata.add(instance);
			 
		}

		weka_traindata.setClassIndex(featureNum);
		
		System.err.println("prodcue"+weka_traindata.numInstances()+" training samples, class "+class_index);
		return weka_traindata;
	}
	//convert Tree instance to weka instance for particualar class
	public static Instances Tree2Weka(List<Tree> trainSample, int class_index)
	{
		System.err.println("working on "+trainSample.size()+" samples, class "+class_index);
		trainSample=BalanceSamples(trainSample,class_index);
		
		return Tree2Weka_(trainSample,class_index);
	}
	
	//Balance CG bias, and length 
	public static List<Tree> BalanceSamples(List<Tree> trainSamples, int class_index)
	{
		System.err.println("BalanceSamples......");
		 int numBin=10;
		EmpiricalDistribution CGdistribution=new EmpiricalDistribution(numBin);
		EmpiricalDistribution Lengthdistribution=new EmpiricalDistribution(numBin);
		ArrayList<Double> CGvalues=new ArrayList<Double>();
		ArrayList<Double> Lenvalues=new ArrayList<Double>();
		ArrayList<Tree> Positie_samples=new ArrayList<Tree>(trainSamples.size()/2);
		ArrayList<Tree> balance_samples=new ArrayList<Tree>(trainSamples.size()/2);
		ArrayList<Double> bgCGvalues=new ArrayList<Double>();
		ArrayList<Double> bgLenvalues=new ArrayList<Double>();
		
		KDTree<Tree> bgTree=new KDTree<Tree>(2);
		Random rand=new Random(0);
		int numBGsamples=0;
		for (int i = 0; i < trainSamples.size(); i++) {
			Tree queryTree = trainSamples.get(i);
			SimpleMatrix all_targets = MotifRNNCoreAnnotations.getGoldClass(queryTree);
			double target=all_targets.get(class_index);
			double cg=0;
			double len=0;
			double[] cglen=new double[2];
			boolean tooLong=false;
			for (LabeledWord leaf : queryTree.labeledYield()) {
				String[] comps = leaf.toString().split("/");
				String motif_name=comps[0];

				if(motif_name.startsWith("CG_content"))
				{
					cg=Double.valueOf(comps[1])+(rand.nextDouble()-0.5)/10000;
					cglen[0]=cg;
				}
				if(motif_name.startsWith("Length_"))
				{
					double tmp=Double.valueOf(comps[1]);
					len=tmp*100+(rand.nextDouble()-0.5)/10000;
					cglen[1]=len;
					if(tmp>Utils.region_length_threshold) 
						tooLong=true;
				}
				
			}
			if(tooLong)
				continue;
			
			if(target>=1)
			{
				Positie_samples.add(queryTree);
				CGvalues.add(cg);
				Lenvalues.add(len);
				
			}
			else if(!Utils.useNoModificationRegionAsBG||all_targets.elementSum()<=0) //must no other histone signal
			{
				numBGsamples+=1;
				bgCGvalues.add(cg);
				bgLenvalues.add(len);
				
				try {
					//record cglen vec, so that later can easy delete from KDtree
					CoreLabel label = (CoreLabel) queryTree.label();
					SimpleMatrix cgVec = new SimpleMatrix(2,1);
					cgVec.set(0, cglen[0]);
					cgVec.set(1, cglen[1]);
					label.set(MotifRNNCoreAnnotations.NodeVector.class, cgVec);
					bgTree.insert(cglen, queryTree);
				} catch (KeySizeException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (KeyDuplicateException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}

		}
		
		
		CGdistribution.load(ArrayUtils.toPrimitive(CGvalues.toArray(new Double[0])));
		Lengthdistribution.load(ArrayUtils.toPrimitive(Lenvalues.toArray(new Double[0])));
		
		ArrayList<Double> newbgCG=new ArrayList<Double>();
		ArrayList<Double> newbgLen=new ArrayList<Double>();
		int num_posSamples=Math.min(Utils.max_train_sample_size/2,Positie_samples.size());
		double positiveSampleRate=(bgTree.size()*1.0)/num_posSamples;  //the reason for 0.5 here, to avoid oversampling from background
		for (int i = 0; i < num_posSamples; i++) {
			if(rand.nextDouble()>positiveSampleRate)
				continue;
			double[] cglen=new double[2];
			cglen[0]=CGdistribution.getNextValue();
			cglen[1]=Lengthdistribution.getNextValue();
			try {
				if(bgTree.size()==0)
					break;
				Tree selTree = bgTree.nearest(cglen);
				balance_samples.add(selTree); //add bg sampl
				balance_samples.add(Positie_samples.get(i)); //add positive sampl
				//delete from KDTree
				double [] cgvec=new double[2];
				SimpleMatrix cglenMat = MotifRNNCoreAnnotations.getNodeVector(selTree);
				cgvec[0]=cglenMat.get(0);
				cgvec[1]=cglenMat.get(1);
				try {
					bgTree.delete(cgvec);
				} catch (KeyMissingException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				for (LabeledWord leaf : selTree.labeledYield()) {
					String[] comps = leaf.toString().split("/");
					String motif_name=comps[0];

					if(motif_name.startsWith("CG_content"))
					{
						double cg=Double.valueOf(comps[1]);
						newbgCG.add(cg);
					}
					if(motif_name.startsWith("Length_"))
					{
						double len=Double.valueOf(comps[1])*100;
						newbgLen.add(len);
					}
					
				}
				
			} catch (KeySizeException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
//		EmpiricalDistribution CGdistribution2=new EmpiricalDistribution(numBin);
//		EmpiricalDistribution Lengthdistribution2=new EmpiricalDistribution(numBin);
//		CGdistribution2.load(ArrayUtils.toPrimitive(newbgCG.toArray(new Double[0])));
//		Lengthdistribution2.load(ArrayUtils.toPrimitive(newbgLen.toArray(new Double[0])));
//		List<SummaryStatistics> CGstat1 = CGdistribution.getBinStats();
//		List<SummaryStatistics> CGstat2 = CGdistribution2.getBinStats();
//		for (int i = 0; i < numBin; i++) {
//			System.out.println(CGstat1.get(i)+"-----------"+CGstat2.get(i));
//		}

		return balance_samples;
	}
	
	public static Instances Fasta2Weka(String inputfile, String flankingFile)
	{
		try {
			Instances ints1=Fasta2Weka(inputfile);
			int kmerlen=Utils.kmerLen;
			Utils.kmerLen=kmerlen;
			Instances ints2=Fasta2Weka(flankingFile);
			Utils.kmerLen=kmerlen;
		      int featureNum=ints1.numAttributes()+ints2.numAttributes()-1;
				FastVector attrList=new FastVector(featureNum);
				for (int i = 0; i < ints2.numAttributes()-1; i++) {
					attrList.addElement(new Attribute("flank_"+ints2.attribute(i).name()));
				}
				for (int i = 0; i < ints1.numAttributes()-1; i++) {
					attrList.addElement(new Attribute(ints1.attribute(i).name()));
				}
				FastVector Tvalues = new FastVector();
				Tvalues.addElement("negative");
				Tvalues.addElement("positive");
				Attribute label = new Attribute("target_value",Tvalues);	
				attrList.addElement(label);
			//usually the flanking is smaller size, because out of chromosome size
				System.err.println("Total number of features: "+(featureNum-1));
			 Instances weka_traindata=new Instances(inputfile, attrList,ints2.numInstances());
			 int i1=0;
			 for (int i = 0; i < Math.min(ints2.numInstances(),ints1.numInstances()); i++) {
				 Instance s1 = ints1.instance(i1);
				 Instance s2 = ints2.instance(i);
				 if(s1.classValue()!=s2.classValue())
				 {
					 i1++;
					 s1 = ints1.instance(i1);
				 }
				 double[] sampleVec=new double[featureNum];
				 double[] temp1 = s2.toDoubleArray();
				 for (int j = 0; j < temp1.length-1; j++) {
					 sampleVec[j]=temp1[j];
				}
				 double[] temp2 = s1.toDoubleArray();
				 for (int j = 0; j < temp2.length-1; j++) {
					 sampleVec[j+temp1.length]=temp2[j];
				}
				 sampleVec[sampleVec.length-1]=s1.classValue();
				 i1++;
				 Instance instance = new Instance(1, sampleVec);
				 weka_traindata.add(instance);
			}
			 weka_traindata.setClassIndex(featureNum-1);
			 return weka_traindata;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return null;
		
	}
	
	public static Instances Fasta2Weka(String inputfile) throws Exception
	{
		System.err.println("convert fasta to weka instance.");
	      BufferedReader br = new BufferedReader(new FileReader(inputfile));
	      String Line="";
	      String seqstr="";
	      
	      int kmerLen=Utils.kmerLen;
	      int kmerNum=(int) Math.pow(4, kmerLen);
	      int featureNum=kmerNum;
//	      if(Utils.enableGapKmer)
//	      {
//	    	  featureNum*=2;
//	      }
	      ////////////build hashmap to record the motif name and feature id////////////////
	      HashMap<Long, Integer> kmerID2FeatureId=new HashMap<Long, Integer>();
	      LinkedList<String> featureName=new LinkedList<String>();
	      for (int i = 0; i <  kmerNum; i++) {
	    	  long kmerId2=Utils.reverseComplement(i, kmerLen);
	    	  if(i<=kmerId2)
	    	  {
	    	  kmerID2FeatureId.put(kmerId2, featureName.size());
	    	  kmerID2FeatureId.put((long) i, featureName.size());
	    	  featureName.add(Utils.longToString(i, kmerLen));
	    	  }
		}
	      if(Utils.enableGapKmer)
	      {
	    	  int ungapFeatNum=kmerNum;
		      for (int i = 0; i <  kmerNum; i++) {
		    	  long kmerId2=Utils.reverseComplement(i, kmerLen);
		    	  if(i<=kmerId2)
		    	  {
		    	  kmerID2FeatureId.put(ungapFeatNum+kmerId2, featureName.size());
		    	  kmerID2FeatureId.put(ungapFeatNum+(long) i, featureName.size());
		    	  featureName.add(Utils.insertGaptoKmer(Utils.longToString(i, kmerLen),Utils.gapLen));
		    	  }
			}
	      }
	      featureNum=featureName.size();
	      
			FastVector attrList=new FastVector(featureNum+1);
			for (int i = 0; i < featureNum; i++) {
//				if(!Utils.enableGapKmer||i<featureNum/2)  //normal k-mer
//					attrList.addElement(new Attribute(Utils.longToString(i, kmerLen)));
//				else
//				{
//						int gid=i-featureNum/2;
//						String ungap_str=Utils.longToString(gid, kmerLen);
//						attrList.addElement(new Attribute(Utils.insertGaptoKmer(ungap_str,Utils.gapLen)));
//				}
				attrList.addElement(new Attribute(featureName.get(i)));
			}
			
			 ///////weight by KmerPrior////////
			double [] kmerPrior_Vec=new double[attrList.size()];
			for (int i = 0; i < kmerPrior_Vec.length; i++) {
				kmerPrior_Vec[i]=1;
			}
			 if(kmerPrior!=null&&kmerPrior.size()>1)
			 {
				 for(String kmer : kmerPrior.keySet())
				 {
					 long kmerId= Utils.charsToLong(kmer.toCharArray(),0,kmerLen);
                    int featid=kmerID2FeatureId.get(kmerId);
                    kmerPrior_Vec[featid]=Math.exp(kmerPrior.get(kmer));
				 }
			 }	
			 //////////////////////////////////
			FastVector Tvalues = new FastVector();
			Tvalues.addElement("negative");
			Tvalues.addElement("positive");
			Attribute label = new Attribute("target_value",Tvalues);
			
			attrList.addElement(label);
		     String seqname="";
		     String lastseqname="";
		 LinkedList<Instance> instList=new LinkedList<Instance>();
		 int numLines=0;
	     while( (Line=br.readLine())!=null)
	     {
	    	 if(Line.length()>0)
	    	 {
	    		 if(Line.charAt(0)=='>')
	    		 {
	    			 numLines+=1;
	    			 seqname=Line.substring(1);
	    			 seqstr=seqstr.replace("N", "");
	    			 seqstr=seqstr.replace("-", "");
	    			 if(seqstr.length()>0)
	    			 {
	    			 char[] charArr=seqstr.toCharArray();
	    			 double[] values = new double[featureNum+1] ;
	    			 
	    			 for (int i = 0; i < charArr.length-kmerLen; i++) {
	    				long kmerId= Utils.charsToLong(charArr,i,i+kmerLen);
//	    				long kmerId2=Utils.reverseComplement(kmerId, kmerLen);
//	    				if(kmerId>kmerId2)
//	    				{
//	    					kmerId=kmerId2;
//	    				}
	    				
	    				int featid=kmerID2FeatureId.get(kmerId);
	    				values[featid]+=1;		
					}
	    			 if(Utils.enableGapKmer)
	    			 {
		    			 for (int i = 0; i < charArr.length-kmerLen-Utils.gapLen; i++) {
		    				 String temp_charArr=seqstr.substring(i, i+kmerLen/2)+seqstr.substring( i+kmerLen/2+Utils.gapLen,i+kmerLen+Utils.gapLen);
			    				long kmerId= Utils.charsToLong(temp_charArr.toCharArray(),0,kmerLen);
//			    				long kmerId2=Utils.reverseComplement(kmerId, kmerLen);
//			    				if(kmerId>kmerId2)
//			    				{
//			    					kmerId=kmerId2;
//			    				}
			    				int featid=kmerID2FeatureId.get(kmerNum+kmerId);
			    				values[featid]+=1;		
//			    				values[featureNum/2+(int) kmerId]+=1;		
							}
	    			 }
	    			 
	    			 ///////weight by KmerPrior////////
	    			 if(kmerPrior!=null&&kmerPrior.size()>1)
	    			 {
	    				for (int i = 0; i < kmerPrior_Vec.length; i++) {
//	    					if(kmerPrior_Vec[i]<1)
//	    						values[i]=0;
	    					values[i]*=kmerPrior_Vec[i];
						}
	    			 }
	    			 
	    			 double sumSQ=0; //use sumSQ instead of sum!!!
	    			 for (int i = 0; i < values.length-1; i++) {
	    				 sumSQ+=values[i]*values[i];
	    			 }
	    			 sumSQ=Math.sqrt(sumSQ);

	    			 if(sumSQ>0)
	    			 //normalized to 1
	    			 for (int i = 0; i < values.length-1; i++) {
	    				 values[i]/=sumSQ;
					}
	    			 

	    			 
	    			 if(lastseqname.contains("Pos"))
	    			 {
	    				 values[featureNum]=1;
	    			 }
	    			 else
	    			 {
	    				 values[featureNum]=0;
	    			 }
	    			 Instance instance = new Instance(1, values);
	    			 instList.add(instance);
	    			 seqstr="";
	    			 }
	    			 lastseqname=seqname;
	    		 }
	    		 else
	    		 {
	    			 seqstr+=Line.trim();
	    		 }
	    	 }
	     }

	     
	     
	   /////////////////  ///the last sequence://///////////////
	     
	     seqstr=seqstr.replace("N", "");
		 seqstr=seqstr.replace("-", "");
		 char[] charArr=seqstr.toCharArray();
		 double[] values = new double[featureNum+1] ;
		 
		 for (int i = 0; i < charArr.length-kmerLen; i++) {
			long kmerId= Utils.charsToLong(charArr,i,i+kmerLen);
//			long kmerId2=Utils.reverseComplement(kmerId, kmerLen);
//			if(kmerId>kmerId2)
//			{
//				kmerId=kmerId2;
//			}
//			values[(int) kmerId]+=1;
			int featid=kmerID2FeatureId.get(kmerId);
			values[featid]+=1;
			
		}
		 if(Utils.enableGapKmer)
		 {
			 for (int i = 0; i < charArr.length-kmerLen-Utils.gapLen; i++) {
				 String temp_charArr=seqstr.substring(i, i+kmerLen/2)+seqstr.substring( i+kmerLen/2+Utils.gapLen,i+kmerLen+Utils.gapLen);
    				long kmerId= Utils.charsToLong(temp_charArr.toCharArray(),0,kmerLen);
//    				long kmerId2=Utils.reverseComplement(kmerId, kmerLen);
//    				if(kmerId>kmerId2)
//    				{
//    					kmerId=kmerId2;
//    				}
//    				values[featureNum/2+(int) kmerId]+=1;	
    				int featid=kmerID2FeatureId.get(kmerNum+kmerId);
    				values[featid]+=1;	
				}
		 }
		 
		 ///////weight by KmerPrior////////
		 if(kmerPrior!=null&&kmerPrior.size()>1)
		 {
			for (int i = 0; i < kmerPrior_Vec.length; i++) {
//				if(kmerPrior_Vec[i]<1)
//					values[i]=0;
				values[i]*=kmerPrior_Vec[i];
			}
		 }
		 
		 double sumSQ=0; //use sumSQ instead of sum!!!
		 for (int i = 0; i < values.length-1; i++) {
			 sumSQ+=values[i]*values[i];
		 }
		 sumSQ=Math.sqrt(sumSQ);
		 if(sumSQ>0)
		 {
			 //normalized to 1
			 for (int i = 0; i < values.length-1; i++) {
				 values[i]/=sumSQ;
			}
		 }

			 
			 if(lastseqname.contains("Pos"))
			 {
				 values[featureNum]=1;
			 }
			 else
			 {
				 values[featureNum]=0;
			 }
			 Instance instance = new Instance(1, values);
			 instList.add(instance);
		 
/////////////////  ///finish the last sequence://///////////////		 
	
	
	          br.close();
	         System.err.println("read number of sequences: "+numLines);
	          Instances weka_traindata=new Instances(inputfile, attrList,Math.min(Utils.max_train_sample_size, instList.size()));
	          Iterator<Instance> iter = instList.iterator();
	          Random rand=new Random(1);
	          double samplerate=Utils.max_train_sample_size*0.99/instList.size();
	          while(iter.hasNext())
	          {
	        	  Instance tmp = iter.next();
	        	  if(rand.nextDouble()<samplerate)
	        	  weka_traindata.add(tmp);
	          }
	          weka_traindata.setClassIndex(featureNum);
	        System.err.println("number of training samples: "+weka_traindata.numInstances());
			return weka_traindata;
		
	}
	

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String inputfile=args[0];
		Options options = new Options();
		options.addOption("i", true, "input training data file");
		options.addOption("threadnum", true, "maximum thread number");
		options.addOption("minMap", true, "minimum mappability (default: 1)");
		
		options.addOption("maxLen", true, "maximum region length allow (default: 2000bp)");
		options.addOption("regCoef", true, "the strength of regularization (default: 1)");
		options.addOption("numFeat", true, "number of top features reported");
		options.addOption("kmerLen", true, "the length of k-mer to use as feature (default 6)");
		options.addOption("kmerPrior", true, "the two-column file specify the K-Mer prior probability");
		options.addOption("maxTrainSize", true, "maximum number of samples used in the training (default: 40000)");
		options.addOption("exploreBias", false, "enable the mode only consider bias in the training process");
		options.addOption("excludeRepeat", false, "ignore Repeat feature in the training process");
		options.addOption("maxScore", false, "use maximum of motif/kmer score instead of sum score"); 
		options.addOption("displayAllFeature", false, "display all the top feature including repeat and bias term, by default only output the top motif/kmer features");
		options.addOption("trainOnly", false, "just build model, no cross-validation");
		options.addOption("useModelFile", true, "use model file to predict the input file");
		options.addOption("flank", true, "fasta file or feature file for the flanking regions");
		options.addOption("gapkmer", false, "use gap-kmer as well");
		options.addOption("RNNtree", false, "generate TreeList object for RNN model");
		
		CommandLineParser parser = new GnuParser();
		CommandLine cmd;
		
		boolean trainOnly=false;
		String modelfile="";
		String flankfile="";
		
		//parsing paramters
		try {
			String appPath=new File(".").getCanonicalPath()+"/";
			cmd = parser.parse( options, args);
			if(cmd.hasOption("i"))
				inputfile=cmd.getOptionValue("i");
			else
			{
				System.err.println("must provide a input file!\n"+options.toString());
				
			}
			
			if(cmd.hasOption("useModelFile"))
			{
				modelfile=cmd.getOptionValue("useModelFile");
				Utils.max_train_sample_size=Integer.MAX_VALUE; //just do prediction, so no limit
			}
			if(cmd.hasOption("flank"))
			{
				flankfile=cmd.getOptionValue("flank");
			}
			if(cmd.hasOption("threadnum"))
				MotifSimpleModel.threadNum=Integer.parseInt(cmd.getOptionValue("threadnum"));
			
			if(cmd.hasOption("minMap"))
				Utils.region_mappability_threshold=Double.parseDouble(cmd.getOptionValue("minMap"));
			
			if(cmd.hasOption("maxLen"))
				Utils.region_length_threshold=Double.parseDouble(cmd.getOptionValue("maxLen"))/10000;
			
			if(cmd.hasOption("numFeat"))
				Utils.numTopFeatures=Integer.parseInt(cmd.getOptionValue("numFeat"));
			
			if(cmd.hasOption("maxTrainSize"))
				Utils.max_train_sample_size=Integer.parseInt(cmd.getOptionValue("maxTrainSize"));
			
			if(cmd.hasOption("kmerLen"))
				Utils.kmerLen=Integer.parseInt(cmd.getOptionValue("kmerLen"));
			
			if(cmd.hasOption("exploreBias"))
				Utils.studyGC_CpG_Repeat=true;
			
			if(cmd.hasOption("excludeRepeat"))
				Utils.excludeRepeat=true;
			
			if(cmd.hasOption("maxScore"))
				Utils.sumScore=false;
			if(cmd.hasOption("kmerPrior"))
			{
				 kmerPrior=Utils.ReadFileToHashMap(cmd.getOptionValue("kmerPrior"));
				
			}
				
			if(cmd.hasOption("trainOnly"))
			{
				trainOnly=true;
				Utils.noCV=true;
			
			}
			if(cmd.hasOption("regCoef"))
				Utils.regCoef=Double.parseDouble(cmd.getOptionValue("regCoef"));
			
			if(cmd.hasOption("gapkmer"))
			{
				
				Utils.enableGapKmer=true;
			}
			
			if(cmd.hasOption("displayAllFeature"))
				Utils.displayOnlyMotif=false;
			
		}catch (ParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp( "SimpleModeler", options );
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp( "SimpleModeler", options );
		}
		
		
		long startTime = System.currentTimeMillis();
		if(inputfile.endsWith("tree"))
		{
//		  List<Tree> trainingTrees = Utils.readTreesWithGoldLabels(inputfile);//.subList(0, 30000);

		  MotifSimpleModel modeler=new MotifSimpleModel(new RNNOptions());
		  MotifSimpleModel.setClassNames(FilenameUtils.getBaseName(inputfile).replace(".tree", "").split("_"));
//		  modeler.train(trainingTrees);
		  
		  
		  modeler.train(inputfile);

		  modeler.saveSerialized(inputfile+".model");
		
		}
		else if(inputfile.endsWith("fa"))
		{
		
		try {
			 MotifSimpleModel modeler=new MotifSimpleModel(new RNNOptions());
			 if(modelfile=="")
			 {	 
					 modeler.train_fastaInput(inputfile,flankfile);
					 modeler.saveSerialized(inputfile+".model");
			 }
			 else
			 {
				 modeler= MotifSimpleModel.loadSerialized(modelfile);
				 modeler.predict_fastaInput(inputfile,flankfile);
			 }
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		}
		else if(inputfile.endsWith("txt"))
		{
			 MotifSimpleModel modeler=new MotifSimpleModel(new RNNOptions());
			 if(modelfile=="")
			 {
			  modeler.train_flatInput(inputfile);
			  modeler.saveSerialized(inputfile+".model");
			 }
			 else
			 {
				 modeler= MotifSimpleModel.loadSerialized(modelfile);
				 modeler.predict_flatInput(inputfile);
				 
			 }
		}
		else
		{
			System.out.println("the input file has to be tree or fa");
		}
		long estimatedTime = System.currentTimeMillis() - startTime;
		System.err.println("time elapse: "+estimatedTime/1000+" sec");
	}

}
