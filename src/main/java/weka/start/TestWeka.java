package weka.start;

import weka.attributeSelection.CfsSubsetEval;
import weka.core.*;

import java.util.ArrayList;
import java.util.List;

public class TestWeka {

    public static void main(String[] args) throws Exception{
        testInstances();
        testAttributes();

    }

    private static void testAttributes() throws Exception{
        List<String> attributesList = new ArrayList();
        attributesList.add("first");
        attributesList.add("second");
        Attribute attribute = new Attribute("positions",attributesList);
        System.out.println(attribute);
        ArrayList<Attribute> newList = new ArrayList<Attribute>();
        newList.add(attribute);
        Instances dataUnlabeled = new Instances("TestInstances", newList,0);
        new CfsSubsetEval().buildEvaluator(dataUnlabeled);

    }

    private static void testInstances() throws Exception{
        //We can load the instances from the csv files etc.
        Instance instance = new SparseInstance(3);
        instance.setValue(0,1);
        instance.setValue(1,3);
        instance.setValue(2,4);


        System.out.println("Started evaluating the model....");
        Instance denseInstance = new DenseInstance(3);
        denseInstance.setValue(0,1);
        denseInstance.setValue(1,3);
        denseInstance.setValue(2,4);

    }
}
