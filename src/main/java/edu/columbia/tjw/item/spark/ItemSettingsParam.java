/*
 * Copyright 2017 tyler.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.columbia.tjw.item.spark;

import edu.columbia.tjw.item.ItemSettings;
import java.io.Serializable;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.util.Identifiable;

/**
 *
 * @author tyler
 */
public class ItemSettingsParam extends Param<ItemSettings>
{
    private static final String GUID = "7cc313e747f68bec5a42979e8a373d24";
    private static final SettingsParamIdentity IDENT = new SettingsParamIdentity();
    private static final long serialVersionUID = 0x736847d3b34dbdebL;
    private static final ItemSettingsParam SINGLETON = new ItemSettingsParam();

    public static ItemSettingsParam singleton()
    {
        return SINGLETON;
    }

    private ItemSettingsParam()
    {
        super(IDENT, "ItemSettings", "A consolidated param to hold basic Item Settings Data.");
    }

    private static final class SettingsParamIdentity implements Identifiable, Serializable
    {
        private static final long serialVersionUID = 0xb59fd19fadb7f134L;

        private SettingsParamIdentity()
        {
            //Do nothing.
        }

        @Override
        public String uid()
        {
            return GUID;
        }

    }

}
