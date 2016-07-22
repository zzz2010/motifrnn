import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import edu.wlu.cs.levy.CG.KeyDuplicateException;
import edu.wlu.cs.levy.CG.KeySizeException;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;


public class BiasUnremovableDemo {

	
	public static Instances FlatString2Weka(String filepath, String class_str)
	{
		BufferedReader br;
		ArrayList<String>trainSample=new ArrayList<String>();
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
				if(target>=1)
				{
					trainSample.add(line);
				}
				else if(!morethan1Class) //must no other histone signal there
				{
					trainSample.add(line);
				}
			}
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		return FlatString2Weka_(trainSample,class_str);
	}
	
	static boolean isTF(String featname)
	{
		return featname.contains("POU5F1")||featname.contains("CTCF")||featname.contains("CUX1");
	}
	public static HashMap<String, Double> sampleExtends(HashMap<String, Double> sample)
	{
		Random rand=new Random(1);
		HashMap<String, Double> sampleRet=new HashMap<String, Double>();
		sampleRet.putAll(sample);
		int n_each=2;
		ArrayList<String> FeatNames=new ArrayList<String>();
		for (String featname :sample.keySet()) {
			if(isTF(featname))
				continue;
			FeatNames.add(featname);
		}
		
		for (Entry<String, Double> feat : sample.entrySet()) {
			//square
			if(isTF(feat.getKey()))
				continue;
			for (int i = 0; i < n_each; i++) {
				String name=feat.getKey()+"_square."+i;
				double temp=Math.pow(feat.getValue()+rand.nextDouble(), 2);
				sampleRet.put(name, temp);
			}
			
			//cube
			for (int i = 0; i < n_each; i++) {
				String name=feat.getKey()+"_cube."+i;
				double temp=Math.pow(feat.getValue()+rand.nextDouble(), 3);
				sampleRet.put(name, temp);
			}
			
			//square root
			for (int i = 0; i < n_each; i++) {
				String name=feat.getKey()+"_sqrt."+i;
				double temp=Math.sqrt(feat.getValue()+rand.nextDouble());
				sampleRet.put(name, temp);
			}
			//ln
			for (int i = 0; i < n_each; i++) {
				String name=feat.getKey()+"_ln."+i;
				double temp=Math.log(feat.getValue()+rand.nextDouble());
				sampleRet.put(name, temp);
			}
			
			//log10
			for (int i = 0; i < n_each; i++) {
				String name=feat.getKey()+"_l0g10."+i;
				double temp=Math.log10(feat.getValue()+rand.nextDouble());
				sampleRet.put(name, temp);
			}
			
			//abs -1
			for (int i = 0; i < n_each; i++) {
				String name=feat.getKey()+"_abs."+i;
				double temp=Math.abs(feat.getValue()+rand.nextDouble()-1);
				sampleRet.put(name, temp);
			}
		}
		
		
		//pairewise feature
		for (int k = 0; k < FeatNames.size()-1; k++) {
			for (int j = k+1; j < FeatNames.size(); j++) {
				String name2=FeatNames.get(k)+"|"+FeatNames.get(j);
				double value=sample.get(FeatNames.get(k))*sample.get(FeatNames.get(j));
				//square
				for (int i = 0; i < n_each; i++) {
					String name=name2+"_square."+i;
					double temp=Math.pow(value+rand.nextDouble(), 2);
					sampleRet.put(name, temp);
				}
				
				//cube
				for (int i = 0; i < n_each; i++) {
					String name=name2+"_cube."+i;
					double temp=Math.pow(value+rand.nextDouble(), 3);
					sampleRet.put(name, temp);
				}
				
				//square root
				for (int i = 0; i < n_each; i++) {
					String name=name2+"_sqrt."+i;
					double temp=Math.sqrt(value+rand.nextDouble());
					sampleRet.put(name, temp);
				}
				//ln
				for (int i = 0; i < n_each; i++) {
					String name=name2+"_ln."+i;
					double temp=Math.log(value+rand.nextDouble());
					sampleRet.put(name, temp);
				}
				
				//log10
				for (int i = 0; i < n_each; i++) {
					String name=name2+"_l0g10."+i;
					double temp=Math.log10(value+rand.nextDouble());
					sampleRet.put(name, temp);
				}
				
				//abs -1
				for (int i = 0; i < n_each; i++) {
					String name=name2+"_abs."+i;
					double temp=Math.abs(value+rand.nextDouble()-1);
					sampleRet.put(name, temp);
				}
			}
		}
		
		return sampleRet;
	}
	
	private static Instances FlatString2Weka_(List<String> trainSample, String class_str)
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
				 boolean cofounder=(motif_name.contains("content")||motif_name.contains("CpG")||motif_name.contains("Length")||motif_name.contains("Mapp")||isTF(motif_name));
				 if(!cofounder)
					 continue;
				 

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
			
			sample=sampleExtends(sample);
			for(String featname : sample.keySet())
			{
				if(!motif_Id.containsKey(featname))
				{
					motif_Id.put(featname, mid);
					mid+=1;
				}
			}
			
			double t=0;
			if(comps.length>4&&comps[4].contains(class_str))
				t=1;
			targets.add(t);
			
			flattenSamples.add(sample);
			if(flattenSamples.size()>5000)
				break;
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
				 if(motifname.contains("content")||motifname.contains("CpG")||motifname.contains("Length")||motifname.contains("Mapp")||motifname.contains("POU5F1"))
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
	/**
	 * @param args
	 */
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String inputfile=args[0];
		Instances data = FlatString2Weka(inputfile, "H3K4me3");
		CSVSaver saver=new CSVSaver();
		saver.setInstances(data);
		try {
			saver.setFile(new File("H3K4me3.csv"));
			saver.writeBatch();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		CfsSubsetEval eval=new CfsSubsetEval();
		GreedyStepwise search =new GreedyStepwise();
//		search.setSearchBackwards(true);
		weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
	    filter.setEvaluator(eval);
	    filter.setSearch(search);
	    try {
			filter.setInputFormat(data);
			Instances data2 = Filter.useFilter(data, filter);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    int maxNumofFeature=30;
	    wekaBinaryClassifierWrapper learner=new wekaBinaryClassifierWrapper(new RandomForest(), maxNumofFeature);
	    learner.LassoSelection2(data, maxNumofFeature);
		  Map<String, Float> sortedFeatureWeight = Utils.sortByValue(learner.FeatureImportance);
		  ArrayList<String> selectedFeature = new ArrayList<String>();
		  for (Entry<String, Float> feat : sortedFeatureWeight.entrySet()) {
			  selectedFeature.add(feat.getKey()+":"+feat.getValue());
		}
		 System.out.println(selectedFeature);
		
	}

}
