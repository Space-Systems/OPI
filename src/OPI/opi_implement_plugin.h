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
#ifndef OPI_PLUGIN_IMPLEMENT_PLUGIN_H
#define OPI_PLUGIN_IMPLEMENT_PLUGIN_H
#include "opi_common.h"
#include "opi_plugininfo.h"
// propagator plugin shortcut
#if WIN32
#define OPI_PLUGIN_EXPORT __declspec( dllexport)
#else
#define OPI_PLUGIN_EXPORT
#endif
#ifdef OPI_DECLARE_PROPAGATOR_PLUGIN
#define OPI_DECLARE_PLUGIN
#define OPI_PLUGIN_TYPE OPI_PROPAGATOR_PLUGIN
#endif
// the plugin implements a cpp propagator
#ifdef OPI_IMPLEMENT_CPP_PROPAGATOR
// enable the auto-declaration
#define OPI_DECLARE_PLUGIN
// set the cpp interface variable
#define OPI_PLUGIN_USES_CPP_INTERFACE true
// implement the interface function
#ifdef __cplusplus
extern "C" {
#endif
OPI_PLUGIN_EXPORT OPI::Propagator* OPI_Plugin_createPropagator(OPI::Host& host)
{
	OPI::Propagator* out = new OPI_IMPLEMENT_CPP_PROPAGATOR(host);
	out->setName(OPI_PLUGIN_NAME);
	out->setAuthor(OPI_PLUGIN_AUTHOR);
	out->setDescription(OPI_PLUGIN_DESC);
	return out;
}
// if not already set, set the plugin type to propagator
#ifndef OPI_PLUGIN_TYPE
#define OPI_PLUGIN_TYPE OPI_PROPAGATOR_PLUGIN
#endif
#ifdef __cplusplus
}
#endif
#endif

#ifdef OPI_DECLARE_COLLISION_DETECTION_PLUGIN
#define OPI_DECLARE_PLUGIN
#define OPI_PLUGIN_TYPE OPI_COLLISION_DETECTION_PLUGIN
#endif
// the plugin uses the cpp plugin interface
#ifdef OPI_IMPLEMENT_CPP_COLLISION_DETECTION
// enable the auto-declaration
#define OPI_DECLARE_PLUGIN
// set the cpp interface variable
#define OPI_PLUGIN_USES_CPP_INTERFACE true
// implement the interface function
#ifdef __cplusplus
extern "C" {
#endif
OPI_PLUGIN_EXPORT OPI::CollisionDetection* OPI_Plugin_createCollisionDetection(OPI::Host& host)
{
	OPI::CollisionDetection* out = new OPI_IMPLEMENT_CPP_COLLISION_DETECTION(host);
	out->setName(OPI_PLUGIN_NAME);
	out->setAuthor(OPI_PLUGIN_AUTHOR);
	out->setDescription(OPI_PLUGIN_DESC);
	return out;
}
// if not already set, set the plugin type to propagator
#ifndef OPI_PLUGIN_TYPE
#define OPI_PLUGIN_TYPE OPI_COLLISION_DETECTION_PLUGIN
#endif
#ifdef __cplusplus
}
#endif
#endif

// c query plugin
#ifdef OPI_DECLARE_DISTANCE_QUERY_PLUGIN
#define OPI_DECLARE_PLUGIN
#define OPI_PLUGIN_TYPE OPI_DISTANCE_QUERY_PLUGIN
#endif
// the plugin implements a cpp query plugin
#ifdef OPI_IMPLEMENT_CPP_DISTANCE_QUERY
#define OPI_DECLARE_PLUGIN
// set the cpp interface variable
#define OPI_PLUGIN_USES_CPP_INTERFACE true
// implement the interface function
#ifdef __cplusplus
extern "C" {
#endif
OPI_PLUGIN_EXPORT OPI::DistanceQuery* OPI_Plugin_createDistanceQuery(OPI::Host& host)
{
	OPI::DistanceQuery* out = new OPI_IMPLEMENT_CPP_DISTANCE_QUERY(host);
	out->setName(OPI_PLUGIN_NAME);
	out->setAuthor(OPI_PLUGIN_AUTHOR);
	out->setDescription(OPI_PLUGIN_DESC);
	return out;
}
// if not already set, set the plugin type to propagator
#ifndef OPI_PLUGIN_TYPE
#define OPI_PLUGIN_TYPE OPI_DISTANCE_QUERY_PLUGIN
#endif
#ifdef __cplusplus
}
#endif
#endif



// declare the plugin
#ifdef OPI_DECLARE_PLUGIN
// by default each plugin uses the c interface
#ifndef OPI_PLUGIN_USES_CPP_INTERFACE
#define OPI_PLUGIN_USES_CPP_INTERFACE false
#endif

// declare c namespace
#ifdef __cplusplus
extern "C" {
#endif
#include <string.h>
// impement the pluginInfo function
OPI_PLUGIN_EXPORT int OPI_Plugin_info(OPI_PluginInfo* info)
{
	info->apiVersionMajor = OPI_API_VERSION_MAJOR;
	info->apiVersionMinor = OPI_API_VERSION_MINOR;
	info->name = OPI_PLUGIN_NAME;
	info->name_len = strlen(OPI_PLUGIN_NAME);
	info->author = OPI_PLUGIN_AUTHOR;
	info->author_len = strlen(OPI_PLUGIN_AUTHOR);
	info->desc = OPI_PLUGIN_DESC;
	info->desc_len = strlen(OPI_PLUGIN_DESC);
	info->versionMajor = OPI_PLUGIN_VERSION_MAJOR;
	info->versionMinor = OPI_PLUGIN_VERSION_MINOR;
	info->versionPatch = OPI_PLUGIN_VERSION_PATCH;
	info->type = OPI_PLUGIN_TYPE;
	info->cppPlugin = OPI_PLUGIN_USES_CPP_INTERFACE;
	return 0;
}
// declare c namespace
#ifdef __cplusplus
}
#endif

#endif
#endif
