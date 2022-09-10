package edu.columbia.tjw.item.spark;

import org.apache.spark.SparkContext;
import org.apache.spark.ml.param.Params;
import org.apache.spark.ml.util.DefaultParamsWriter;
import org.apache.spark.ml.util.MLWriter;
import org.json4s.JsonAST;

import java.io.*;
import java.util.zip.GZIPOutputStream;

public class ModelWriterSerializable<T extends Params> extends MLWriter implements Serializable
{
    private final T _underlying;

    public ModelWriterSerializable(T underlying_)
    {
        _underlying = underlying_;
    }


    @Override
    public void saveImpl(final String path_)
    {
        SparkContext sc = SparkContext.getOrCreate();

        final scala.Option<JsonAST.JObject> noneA = scala.Option.apply(null);
        final scala.Option<JsonAST.JValue> noneB = scala.Option.apply(null);

        // This is needed to make the pipeline logic happy when loading this stage.
        // We won't read it ourselves when deserializing.
        DefaultParamsWriter.saveMetadata(_underlying, path_, sc, noneA, noneB);

        // The actual contents of this thing go here. This is the only thing we will read.
        final File dataPath = new File(path_, "data");

        try (final FileOutputStream fout = new FileOutputStream(dataPath);
             final GZIPOutputStream zipOut = new GZIPOutputStream(fout);
             final ObjectOutputStream oOut = new ObjectOutputStream(zipOut);)
        {
            oOut.writeObject(_underlying);
        }
        catch (IOException e)
        {
            throw new RuntimeException("Unable to read data: " + path_, e);
        }
    }
}
