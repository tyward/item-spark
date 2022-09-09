package edu.columbia.tjw.item.spark;

import org.apache.spark.ml.util.MLReader;

import java.io.*;
import java.util.zip.GZIPInputStream;

public class ModelReaderSerializable<T extends Serializable> extends MLReader<T>
{

    @Override
    public T load(final String path_)
    {
        final File dataPath = new File(path_, "data");

        try (final FileInputStream fIn = new FileInputStream(dataPath);
             final GZIPInputStream zipIn = new GZIPInputStream(fIn);
             final ObjectInputStream oIn = new ObjectInputStream(zipIn))
        {
            return (T) oIn.readObject();
        }
        catch (ClassNotFoundException e)
        {
            throw new RuntimeException("Unable to load unknown class.", e);
        }
        catch (IOException e)
        {
            throw new RuntimeException("Unable to read data: " + path_, e);
        }
    }
}
