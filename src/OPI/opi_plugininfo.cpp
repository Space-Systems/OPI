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
