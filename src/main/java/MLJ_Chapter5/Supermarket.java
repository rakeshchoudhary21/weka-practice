package MLJ_Chapter5; /**
 * Chapter 5: Market basket analysis 
 * http://kdd.org/kdd-cup/view/kdd-cup-2009
 * 
 * @author Bostjan Kaluza, http://bostjankaluza.net
 */

import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.Instances;
import weka.associations.Apriori;
import weka.associations.FPGrowth;


public class Supermarket {

	private static final String PATH_TO_DATA = "/Users/r0c0334/Desktop/Weka/weka-practice/src/main/resources/";

	public static void main(String args[]) throws Exception {
		// load data
		Instances data = new Instances(new BufferedReader(new FileReader(PATH_TO_DATA+"data/supermarket.arff")));
		// build model
		Apriori model = new Apriori();
		model.buildAssociations(data);
		System.out.println(model);
		
		FPGrowth fpgModel = new FPGrowth();
		fpgModel.buildAssociations(data);
		System.out.println(fpgModel);
		
		
	}
}
