/* OPI: Orbital Propagation Interface
 * Copyright (C) 2014 Institute of Aerospace Systems, TU Braunschweig, All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.
 */
#include "opi_plugininfo.h"
#include <iostream>
// this file contains mostly helper functions required by fortran plugins
extern "C"
{

void OPI_PluginInfo_init(OPI::PluginInfo* info, int api_major, int api_minor,
												 int plugin_major, int plugin_minor, int plugin_patch,
												 int type)
{
	info->apiVersionMajor = api_major;
	info->apiVersionMinor = api_minor;
	info->cppPlugin = false;
	info->versionMajor = plugin_major;
	info->versionMinor = plugin_minor;
	info->versionPatch = plugin_patch;
	info->type = type;
}
void OPI_PluginInfo_setName(OPI::PluginInfo* info, const char* name, int len)
{
	info->name = name;
	info->name_len = len;
}

void OPI_PluginInfo_setAuthor(OPI::PluginInfo* info, const char* name, int len)
{
	info->author = name;
	info->author_len = len;
}


void OPI_PluginInfo_setDescription(OPI::PluginInfo* info, const char* name, int len)
{
	info->desc = name;
	info->desc_len = len;
}


}
