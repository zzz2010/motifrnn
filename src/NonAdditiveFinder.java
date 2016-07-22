import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.ejml.simple.SimpleMatrix;


public class NonAdditiveFinder {
	public static void main(String[] args) {
		
		  String modelPath = null;
		for (int argIndex = 0; argIndex < args.length; ) {
			
			if (args[argIndex].equalsIgnoreCase("-model")) {  //previous saved model, can use to predict testing data
		        modelPath = args[argIndex + 1];
		        argIndex += 2;
		      } 
		}
		
		MotifSentimentModel model=MotifSentimentModel.loadSerialized(modelPath);
		 Map<String, Float> word_imp = model.GetAllWordsImportance();
		//nonadditive pair effect
		HashMap<String, Float> pairEnrichment=new HashMap<String, Float>();
		String[] wordlists=model.wordVectors.keySet().toArray(new String[0]);
		for (int i = 0; i < wordlists.length; i++) {
			String word1=wordlists[i];
			SimpleMatrix pred1 = model.SingleWordPrediction(word1);
			int maxI1 = Utils.argMax(pred1);
			for (int j = i; j < wordlists.length; j++)
			{
				String word2=wordlists[j];
				SimpleMatrix pred2 = model.SingleWordPrediction(word2);
				int maxI2 = Utils.argMax(pred2);
				SimpleMatrix pred3=model.SingleWordPrediction("(1 "+word1+") "+"(1 "+word2+")");
				int maxI3 = Utils.argMax(pred3);
				if(maxI3==maxI2||maxI3==maxI1)
					continue;
				String outstr=word1+"|"+Utils.Vector2String(pred1);
				outstr+=" + "+word2+"|"+Utils.Vector2String(pred2);
				outstr+=" -> "+Utils.Vector2String(pred3);
				float enrichment=(float) Math.min(Math.abs(pred3.get(maxI3)-pred2.get(maxI3)),Math.abs(pred3.get(maxI3)-pred1.get(maxI3)));
				pairEnrichment.put(outstr, enrichment);
			}
		}
		
		Map<String, Float> sortedPairs = Utils.sortByAbsValue(pairEnrichment);
		int top=10;
		int ii=0;		
		 for (Entry<String, Float> pair : sortedPairs.entrySet()) {
			 System.out.println(pair.getKey());
			 ii+=1;
			 if(ii>top)
				 break;
		 }
		 
		//nonadditive triple order effect
		 System.out.println("===========================================================");
		 HashMap<String, Float> TripleEnrichment=new HashMap<String, Float>();
			for (int i = 0; i < wordlists.length-1; i++) {
				String word1=wordlists[i];
				SimpleMatrix pred1 = model.SingleWordPrediction(word1);
				int maxI1 = Utils.argMax(pred1);
				for (int j = i; j < wordlists.length-1; j++)
				{
					String word2=wordlists[j];
					SimpleMatrix pred2 = model.SingleWordPrediction(word2);
					int maxI2 = Utils.argMax(pred2);
					for (int k = j+1; k < wordlists.length; k++)
					{
						String word3=wordlists[k];
						SimpleMatrix pred3 = model.SingleWordPrediction(word3);
						int maxI3 = Utils.argMax(pred3);
						
						SimpleMatrix pred_order1=model.SingleWordPrediction("((1 "+word1+") "+"(1 "+word2+")) "+"(1 "+word3+")");
						
						SimpleMatrix pred_order2=model.SingleWordPrediction("((1 "+word1+") "+"(1 "+word3+")) "+"(1 "+word2+")");
						
						float enrichment=(float) pred_order1.minus(pred_order2).elementMaxAbs();
						String outstr=word1+","+word2+","+word3+"|"+Utils.Vector2String(pred_order1);
						outstr+="   "+word1+","+word3+","+word2+"|"+Utils.Vector2String(pred_order2);
						TripleEnrichment.put(outstr, enrichment);
					}
					
				}
			}
			Map<String, Float> sortedTriples = Utils.sortByAbsValue(TripleEnrichment);
			top=30;
			 ii=0;		
			 for (Entry<String, Float> triple: sortedTriples.entrySet()) {
				 System.out.println(triple.getKey());
				 ii+=1;
				 if(ii>top)
					 break;
			 }
			
		 
		
	}

}
