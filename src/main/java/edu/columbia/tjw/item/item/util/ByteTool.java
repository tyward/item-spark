/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.columbia.tjw.item.util;

/**
 *
 * @author tyler
 */
public final class ByteTool
{
    private static final char[] HEX_CHARS =
    {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'
    };

    private ByteTool()
    {
    }

    private static long fromBytes(final byte[] input_, final int offset_, final int byteCount_)
    {
        if ((byteCount_ < 1) || (byteCount_ > 8))
        {
            throw new IllegalArgumentException("Invalid byte count: " + byteCount_);
        }

        long output = 0;

        for (int i = 0; i < byteCount_; i++)
        {
            final int next = input_[offset_ + i];
            final int converted = (0xFF & next);
            output = output << 8;
            output += converted;
        }

        return output;
    }

    private static void toBytes(final long input_, final int offset_, final byte[] output_, final int byteCount_)
    {
        long workspace = input_;

        for (int i = 0; i < byteCount_; i++)
        {
            final byte thisByte = (byte) (workspace & 0xFFL);
            workspace = workspace >> 8;
            output_[offset_ + ((byteCount_ - 1) - i)] = thisByte;
        }
    }

    public static byte[] longToBytes(final long input_)
    {
        final byte[] output = new byte[8];
        longToBytes(input_, 0, output);
        return output;
    }

    public static void longToBytes(final long input_, final int offset_, final byte[] output_)
    {
        toBytes(input_, offset_, output_, 8);
    }

    public static long bytesToLong(final byte[] input_, final int offset_)
    {
        final long output = fromBytes(input_, offset_, 8);
        return output;
    }

    public static byte[] bytesFromHex(final String input_)
    {
        final int length = input_.length();

        if (length % 2 != 0)
        {
            throw new IllegalArgumentException("Length must be even.");
        }

        final int outputLength = length / 2;

        final byte[] output = new byte[outputLength];

        for (int i = 0; i < outputLength; i++)
        {
            final byte high = (byte) (decodeOne(input_.charAt(2*i)) << 4);
            final byte low = decodeOne(input_.charAt((2*i) + 1));
            final byte next = (byte) (high | low);
            output[i] = next;
        }

        return output;
    }

    public static byte decodeOne(final char input_)
    {
        switch (input_)
        {
            case '0':
                return 0;
            case '1':
                return 1;
            case '2':
                return 2;
            case '3':
                return 3;
            case '4':
                return 4;
            case '5':
                return 5;
            case '6':
                return 6;
            case '7':
                return 7;
            case '8':
                return 8;
            case '9':
                return 9;
            case 'a':
            case 'A':
                return 10;
            case 'b':
            case 'B':
                return 11;
            case 'c':
            case 'C':
                return 12;
            case 'd':
            case 'D':
                return 13;
            case 'e':
            case 'E':
                return 14;
            case 'f':
            case 'F':
                return 15;
            default:
                throw new IllegalArgumentException("Unsupported: " + input_);
        }

    }

    public static String bytesToHex(final byte[] input_)
    {
        char[] chars = new char[input_.length * 2];

        for (int i = 0; i < input_.length; i++)
        {
            final int next = input_[i] & 0xFF;
            final int high = (next >> 4) & 0x0F;
            final int low = next & 0x0F;
            chars[2 * i] = HEX_CHARS[high];
            chars[2 * i + 1] = HEX_CHARS[low];
        }

        final String output = new String(chars);
        return output;
    }

}
