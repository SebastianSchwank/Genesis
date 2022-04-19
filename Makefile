#############################################################################
# Makefile for building: ANN-Sofie_GUI
# Generated by qmake (3.1) (Qt 5.9.9)
# Project:  ANN-Sofie_GUI.pro
# Template: app
# Command: D:\Qt_new\5.9.9\mingw53_32\bin\qmake.exe -o Makefile ANN-Sofie_GUI.pro -spec win32-g++
#############################################################################

MAKEFILE      = Makefile

first: release
install: release-install
uninstall: release-uninstall
QMAKE         = D:\Qt_new\5.9.9\mingw53_32\bin\qmake.exe
DEL_FILE      = del
CHK_DIR_EXISTS= if not exist
MKDIR         = mkdir
COPY          = copy /y
COPY_FILE     = copy /y
COPY_DIR      = xcopy /s /q /y /i
INSTALL_FILE  = copy /y
INSTALL_PROGRAM = copy /y
INSTALL_DIR   = xcopy /s /q /y /i
QINSTALL      = D:\Qt_new\5.9.9\mingw53_32\bin\qmake.exe -install qinstall
QINSTALL_PROGRAM = D:\Qt_new\5.9.9\mingw53_32\bin\qmake.exe -install qinstall -exe
DEL_FILE      = del
SYMLINK       = $(QMAKE) -install ln -f -s
DEL_DIR       = rmdir
MOVE          = move
SUBTARGETS    =  \
		release \
		debug


release: FORCE
	$(MAKE) -f $(MAKEFILE).Release
release-make_first: FORCE
	$(MAKE) -f $(MAKEFILE).Release 
release-all: FORCE
	$(MAKE) -f $(MAKEFILE).Release all
release-clean: FORCE
	$(MAKE) -f $(MAKEFILE).Release clean
release-distclean: FORCE
	$(MAKE) -f $(MAKEFILE).Release distclean
release-install: FORCE
	$(MAKE) -f $(MAKEFILE).Release install
release-uninstall: FORCE
	$(MAKE) -f $(MAKEFILE).Release uninstall
debug: FORCE
	$(MAKE) -f $(MAKEFILE).Debug
debug-make_first: FORCE
	$(MAKE) -f $(MAKEFILE).Debug 
debug-all: FORCE
	$(MAKE) -f $(MAKEFILE).Debug all
debug-clean: FORCE
	$(MAKE) -f $(MAKEFILE).Debug clean
debug-distclean: FORCE
	$(MAKE) -f $(MAKEFILE).Debug distclean
debug-install: FORCE
	$(MAKE) -f $(MAKEFILE).Debug install
debug-uninstall: FORCE
	$(MAKE) -f $(MAKEFILE).Debug uninstall

Makefile: ANN-Sofie_GUI.pro ../Qt_new/5.9.9/mingw53_32/mkspecs/win32-g++/qmake.conf ../Qt_new/5.9.9/mingw53_32/mkspecs/features/spec_pre.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/qdevice.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/device_config.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/common/sanitize.conf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/common/gcc-base.conf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/common/g++-base.conf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/common/angle.conf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/qconfig.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3danimation.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3danimation_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dcore.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dcore_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dextras.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dextras_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dinput.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dinput_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dlogic.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dlogic_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquick.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquick_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickanimation.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickanimation_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickextras.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickextras_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickinput.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickinput_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickrender.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickrender_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickscene2d.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickscene2d_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3drender.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3drender_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_accessibility_support_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_axbase.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_axbase_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_axcontainer.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_axcontainer_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_axserver.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_axserver_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_bluetooth.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_bluetooth_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_bootstrap_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_charts.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_charts_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_concurrent.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_concurrent_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_core.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_core_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_datavisualization.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_datavisualization_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_dbus.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_dbus_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_designer.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_designer_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_designercomponents_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_devicediscovery_support_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_egl_support_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_eventdispatcher_support_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_fb_support_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_fontdatabase_support_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_gamepad.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_gamepad_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_gui.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_gui_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_help.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_help_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_location.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_location_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_multimedia.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_multimedia_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_multimediawidgets.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_multimediawidgets_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_network.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_network_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_networkauth.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_networkauth_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_nfc.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_nfc_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_opengl.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_opengl_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_openglextensions.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_openglextensions_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_packetprotocol_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_platformcompositor_support_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_positioning.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_positioning_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_printsupport.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_printsupport_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_purchasing.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_purchasing_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_qml.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_qml_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_qmldebug_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_qmldevtools_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_qmltest.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_qmltest_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_qtmultimediaquicktools_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quick.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quick_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quickcontrols2.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quickcontrols2_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quickparticles_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quicktemplates2_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quickwidgets.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quickwidgets_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_remoteobjects.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_remoteobjects_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_repparser.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_repparser_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_script.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_script_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_scripttools.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_scripttools_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_scxml.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_scxml_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_sensors.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_sensors_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_serialbus.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_serialbus_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_serialport.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_serialport_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_sql.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_sql_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_svg.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_svg_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_testlib.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_testlib_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_texttospeech.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_texttospeech_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_theme_support_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_uiplugin.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_uitools.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_uitools_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_webchannel.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_webchannel_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_websockets.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_websockets_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_widgets.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_widgets_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_winextras.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_winextras_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_xml.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_xml_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_xmlpatterns.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_xmlpatterns_private.pri \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/qt_functions.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/qt_config.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/win32-g++/qmake.conf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/spec_post.prf \
		.qmake.stash \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/exclusive_builds.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/toolchain.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/default_pre.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/win32/default_pre.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/resolve_config.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/exclusive_builds_post.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/default_post.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/precompile_header.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/warn_on.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/qt.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/resources.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/moc.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/win32/opengl.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/uic.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/qmake_use.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/file_copies.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/win32/windows.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/testcase_targets.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/exceptions.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/yacc.prf \
		../Qt_new/5.9.9/mingw53_32/mkspecs/features/lex.prf \
		ANN-Sofie_GUI.pro \
		../Qt_new/5.9.9/mingw53_32/lib/qtmain.prl \
		../Qt_new/5.9.9/mingw53_32/lib/Qt5Widgets.prl \
		../Qt_new/5.9.9/mingw53_32/lib/Qt5Gui.prl \
		../Qt_new/5.9.9/mingw53_32/lib/Qt5Core.prl
	$(QMAKE) -o Makefile ANN-Sofie_GUI.pro -spec win32-g++
../Qt_new/5.9.9/mingw53_32/mkspecs/features/spec_pre.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/qdevice.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/device_config.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/common/sanitize.conf:
../Qt_new/5.9.9/mingw53_32/mkspecs/common/gcc-base.conf:
../Qt_new/5.9.9/mingw53_32/mkspecs/common/g++-base.conf:
../Qt_new/5.9.9/mingw53_32/mkspecs/common/angle.conf:
../Qt_new/5.9.9/mingw53_32/mkspecs/qconfig.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3danimation.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3danimation_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dcore.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dcore_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dextras.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dextras_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dinput.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dinput_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dlogic.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dlogic_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquick.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquick_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickanimation.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickanimation_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickextras.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickextras_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickinput.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickinput_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickrender.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickrender_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickscene2d.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3dquickscene2d_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3drender.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_3drender_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_accessibility_support_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_axbase.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_axbase_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_axcontainer.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_axcontainer_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_axserver.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_axserver_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_bluetooth.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_bluetooth_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_bootstrap_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_charts.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_charts_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_concurrent.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_concurrent_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_core.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_core_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_datavisualization.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_datavisualization_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_dbus.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_dbus_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_designer.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_designer_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_designercomponents_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_devicediscovery_support_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_egl_support_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_eventdispatcher_support_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_fb_support_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_fontdatabase_support_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_gamepad.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_gamepad_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_gui.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_gui_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_help.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_help_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_location.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_location_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_multimedia.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_multimedia_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_multimediawidgets.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_multimediawidgets_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_network.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_network_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_networkauth.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_networkauth_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_nfc.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_nfc_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_opengl.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_opengl_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_openglextensions.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_openglextensions_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_packetprotocol_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_platformcompositor_support_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_positioning.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_positioning_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_printsupport.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_printsupport_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_purchasing.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_purchasing_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_qml.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_qml_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_qmldebug_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_qmldevtools_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_qmltest.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_qmltest_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_qtmultimediaquicktools_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quick.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quick_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quickcontrols2.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quickcontrols2_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quickparticles_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quicktemplates2_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quickwidgets.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_quickwidgets_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_remoteobjects.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_remoteobjects_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_repparser.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_repparser_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_script.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_script_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_scripttools.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_scripttools_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_scxml.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_scxml_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_sensors.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_sensors_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_serialbus.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_serialbus_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_serialport.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_serialport_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_sql.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_sql_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_svg.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_svg_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_testlib.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_testlib_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_texttospeech.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_texttospeech_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_theme_support_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_uiplugin.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_uitools.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_uitools_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_webchannel.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_webchannel_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_websockets.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_websockets_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_widgets.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_widgets_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_winextras.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_winextras_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_xml.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_xml_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_xmlpatterns.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/modules/qt_lib_xmlpatterns_private.pri:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/qt_functions.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/qt_config.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/win32-g++/qmake.conf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/spec_post.prf:
.qmake.stash:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/exclusive_builds.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/toolchain.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/default_pre.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/win32/default_pre.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/resolve_config.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/exclusive_builds_post.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/default_post.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/precompile_header.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/warn_on.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/qt.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/resources.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/moc.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/win32/opengl.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/uic.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/qmake_use.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/file_copies.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/win32/windows.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/testcase_targets.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/exceptions.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/yacc.prf:
../Qt_new/5.9.9/mingw53_32/mkspecs/features/lex.prf:
ANN-Sofie_GUI.pro:
../Qt_new/5.9.9/mingw53_32/lib/qtmain.prl:
../Qt_new/5.9.9/mingw53_32/lib/Qt5Widgets.prl:
../Qt_new/5.9.9/mingw53_32/lib/Qt5Gui.prl:
../Qt_new/5.9.9/mingw53_32/lib/Qt5Core.prl:
qmake: FORCE
	@$(QMAKE) -o Makefile ANN-Sofie_GUI.pro -spec win32-g++

qmake_all: FORCE

make_first: release-make_first debug-make_first  FORCE
all: release-all debug-all  FORCE
clean: release-clean debug-clean  FORCE
distclean: release-distclean debug-distclean  FORCE
	-$(DEL_FILE) Makefile
	-$(DEL_FILE) .qmake.stash

release-mocclean:
	$(MAKE) -f $(MAKEFILE).Release mocclean
debug-mocclean:
	$(MAKE) -f $(MAKEFILE).Debug mocclean
mocclean: release-mocclean debug-mocclean

release-mocables:
	$(MAKE) -f $(MAKEFILE).Release mocables
debug-mocables:
	$(MAKE) -f $(MAKEFILE).Debug mocables
mocables: release-mocables debug-mocables

check: first

benchmark: first
FORCE:

$(MAKEFILE).Release: Makefile
$(MAKEFILE).Debug: Makefile
