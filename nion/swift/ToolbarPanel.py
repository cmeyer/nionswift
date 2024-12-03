from __future__ import annotations

# standard libraries
import gettext
import pathlib
import typing

# third party libraries
import numpy

# local libraries
from nion.swift import DocumentController
from nion.swift import EntityBrowser
from nion.swift import Panel
from nion.swift.model import Feature
from nion.swift.model import Profile
from nion.ui import CanvasItem
from nion.ui import Declarative
from nion.ui import DrawingContext
from nion.ui import UserInterface
from nion.ui import Window
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import ListModel
from nion.utils import Model
from nion.utils import Registry

if typing.TYPE_CHECKING:
    from nion.swift.model import Persistence

_ = gettext.gettext


class ToolModeToolbarWidget(Declarative.Handler):
    toolbar_widget_id = "nion.swift.toolbar-widget.tool-mode"
    toolbar_widget_title = _("Tools")

    def __init__(self, *, document_controller: DocumentController.DocumentController, **kwargs: typing.Any) -> None:
        super().__init__()

        self.radio_button_value: Model.PropertyModel[int] = Model.PropertyModel(0)

        u = Declarative.DeclarativeUI()

        top_row_items = list()
        bottom_row_items = list()
        modes = list()

        tool_actions = list()

        for action in Window.actions.values():
            if action.action_id.startswith("window.set_tool_mode"):
                tool_actions.append(typing.cast(DocumentController.SetToolModeAction, action))

        for i, tool_action in enumerate(tool_actions):
            tool_id = tool_action.tool_mode
            icon_png = tool_action.tool_icon
            tool_tip = tool_action.tool_tip
            key_shortcut = Window.action_shortcuts.get(tool_action.action_id, dict()).get("display_panel", None)
            if key_shortcut:
                tool_tip += f" ({key_shortcut})"
            modes.append(tool_id)
            assert icon_png is not None
            icon_data = CanvasItem.load_rgba_data_from_bytes(icon_png)
            icon_property = "icon_" + tool_id
            setattr(self, icon_property, icon_data)
            radio_button = u.create_radio_button(icon=f"@binding({icon_property})", value=i,
                                                 group_value="@binding(radio_button_value.value)", width=32, height=24,
                                                 tool_tip=tool_tip)
            if i % 2 == 0:
                top_row_items.append(radio_button)
            else:
                bottom_row_items.append(radio_button)

        if len(tool_actions) % 2 == 1:
            bottom_row_items.append(u.create_spacing(32))

        top_row = u.create_row(*top_row_items)
        bottom_row = u.create_row(*bottom_row_items)

        self.ui_view = u.create_row(u.create_column(u.create_spacing(4), top_row, bottom_row, u.create_spacing(4), u.create_stretch()))

        self.radio_button_value.value = modes.index(document_controller.tool_mode)

        def tool_mode_changed(tool_mode: str) -> None:
            self.radio_button_value.value = modes.index(tool_mode)

        self.__tool_mode_changed_event_listener = document_controller.tool_mode_changed_event.listen(tool_mode_changed)

        tool_mode_changed(document_controller.tool_mode)

        def radio_button_changed(property: str) -> None:
            if property == "value":
                mode_index = self.radio_button_value.value
                if mode_index is not None:
                    document_controller.tool_mode = modes[mode_index]

        self.__radio_button_value_listener = self.radio_button_value.property_changed_event.listen(radio_button_changed)

    def close(self) -> None:
        self.__tool_mode_changed_event_listener.close()
        self.__tool_mode_changed_event_listener = typing.cast(Event.EventListener, None)
        self.__radio_button_value_listener.close()
        self.__radio_button_value_listener = typing.cast(Event.EventListener, None)
        super().close()


class ActionTableToolbarWidget(Declarative.Handler):

    def __init__(self, actions: typing.Sequence[Window.Action], document_controller: DocumentController.DocumentController, **kwargs: typing.Any) -> None:
        super().__init__()
        self.__document_controller = document_controller
        u = Declarative.DeclarativeUI()
        top_row = [self.__create_action_button(actions[i]) for i in range(0, len(actions), 2)]
        bottom_row = [self.__create_action_button(actions[i]) for i in range(1, len(actions), 2)]
        self.ui_view = u.create_column(u.create_spacing(4),
                                       u.create_row(*top_row),
                                       u.create_row(*bottom_row),
                                       u.create_spacing(4), u.create_stretch())

    def __create_action_button(self, action: Window.Action) -> Declarative.UIDescription:
        action_id = action.action_id
        action_identifier = action_id.replace(".", "_")
        icon_png = getattr(action, "action_command_icon_png", None)
        if icon_png is not None:
            icon_data = CanvasItem.load_rgba_data_from_bytes(icon_png)
        else:
            icon_data = numpy.full((48, 64), 0x00FFFFFF, dtype=numpy.uint32)
            icon_data[8:40, 8:56] = 0xFFC0C0C0
        icon_property = "icon_" + action_identifier
        setattr(self, icon_property, icon_data)
        tool_tip = getattr(action, "action_tool_tip", getattr(action, "action_name", None))
        key_shortcut = Window.action_shortcuts.get(action_id, dict()).get("display_panel", None)
        if tool_tip and key_shortcut:
            tool_tip += f" ({key_shortcut})"
        u = Declarative.DeclarativeUI()
        perform_function = "perform_" + action_identifier
        def perform_action(widget: UserInterface.Widget) -> None:
            self.__document_controller.perform_action(action_id)
        setattr(self, perform_function, perform_action)
        return u.create_image(image=f"@binding({icon_property})", height=24, width=32, on_clicked=f"{perform_function}", tool_tip=tool_tip)


class RasterZoomToolbarWidget(ActionTableToolbarWidget):
    toolbar_widget_id = "nion.swift.toolbar-widget.raster-zoom"
    toolbar_widget_title = _("Zoom")  # ideally "Raster Zoom" but that makes the title wider than the controls

    def __init__(self, *, document_controller: DocumentController.DocumentController, **kwargs: typing.Any):
        super().__init__(
            [
                Window.actions["raster_display.fit_view"],
                Window.actions["raster_display.1_view"],
                Window.actions["raster_display.fill_view"],
                Window.actions["raster_display.2_view"],
            ],
            document_controller,
            **kwargs
        )


class WorkspaceToolbarWidget(ActionTableToolbarWidget):
    toolbar_widget_id = "nion.swift.toolbar-widget.workspace"
    toolbar_widget_title = _("Workspace")

    def __init__(self, *, document_controller: DocumentController.DocumentController, **kwargs: typing.Any):
        super().__init__(
            [
                Window.actions["workspace.split_horizontal"],
                Window.actions["workspace.split_vertical"],
                Window.actions["workspace.split_2x2"],
                Window.actions["workspace.split_3x2"],
                Window.actions["workspace.split_3x3"],
                Window.actions["workspace.split_4x3"],
                Window.actions["workspace.split_4x4"],
                Window.actions["workspace.split_5x4"],
                Window.actions["display_panel.select_siblings"],
                Window.actions["display_panel.clear"],
                Window.actions["workspace.1x1"],
                Window.actions["display_panel.close"],
            ],
            document_controller,
            **kwargs
        )


class CommandsToolbarWidget(Declarative.Handler):
    toolbar_widget_id = "nion.swift.toolbar-widget.commands"
    toolbar_widget_title = _("Commands")

    def __init__(self, *, document_controller: DocumentController.DocumentController, **kwargs: typing.Any):
        super().__init__()
        self.__document_controller = document_controller

        app = typing.cast(typing.Any, document_controller.app)  # trick typing
        profile: typing.Optional[Profile.Profile] = app._profile if app else None

        # the profile may or may not be present, so we need to handle that case. not present during tests.
        self.action_commands = ListModel.ObservedListModel(profile, "action_commands") if profile else ListModel.ListModel()

        u = Declarative.DeclarativeUI()

        self.ui_view = u.create_row(
            u.create_row(items="action_commands.items", item_component_id="action-command-component", margin=8),
            u.create_column(u.create_stretch(), u.create_push_button(text=_("\N{GEAR}"), style="minimal", width=24, on_clicked="add_action"), u.create_stretch()),
            u.create_stretch()
        )

    def add_action(self, widget: UserInterface.Widget) -> None:
        self.__document_controller.perform_action("application.open_toolbar_command_dialog")

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        if component_id == "action-command-component":
            action_command = typing.cast(Profile.ActionCommand, item)

            u = Declarative.DeclarativeUI()

            class Handler(Declarative.Handler):
                def __init__(self, document_controller: DocumentController.DocumentController, action_command: Profile.ActionCommand) -> None:
                    super().__init__()
                    self.__document_controller = document_controller
                    self.action_command = action_command
                    self.__action_command_listener = action_command.property_changed_event.listen(self.__action_command_changed)
                    self.ui_view = u.create_column(u.create_stretch(), u.create_push_button(text="@binding(action_command.title)", on_clicked="perform", tool_tip="@binding(tool_tip)", style="minimal"), u.create_stretch())

                def __action_command_changed(self, property_name: str) -> None:
                    if property_name == "tool_tip":
                        self.property_changed_event.fire("tool_tip")

                @property
                def tool_tip(self) -> str:
                    action = self.action_command.window_action
                    tool_tip = getattr(action, "action_tool_tip", getattr(action, "action_name", _("Action")))
                    tool_tip = self.action_command.tool_tip or tool_tip
                    key_shortcut = Window.action_shortcuts.get(action.action_id, dict()).get("display_panel", None) if action and action.action_id else None
                    if tool_tip and key_shortcut:
                        tool_tip += f" ({key_shortcut})"
                    return tool_tip

                def perform(self, widget: Declarative.UIWidget) -> None:
                    self.action_command.perform(self.__document_controller)

            return Handler(self.__document_controller, action_command)
        return None


Registry.register_component(ToolModeToolbarWidget, {"toolbar-widget"})
Registry.register_component(RasterZoomToolbarWidget, {"toolbar-widget"})
Registry.register_component(WorkspaceToolbarWidget, {"toolbar-widget"})
Registry.register_component(CommandsToolbarWidget, {"toolbar-widget"})


class ToolbarPanel(Panel.Panel):

    def __init__(self, document_controller: DocumentController.DocumentController, panel_id: str, properties: Persistence.PersistentDictType) -> None:
        super().__init__(document_controller, panel_id, _("Toolbar"))

        self.__component_registered_listener = Registry.listen_component_registered_event(self.__component_registered)

        self.widget = self.ui.create_column_widget()

        # note: "maximum" here means the size hint is maximum and the widget can be smaller. Qt layout is atrocious.
        self.__toolbar_widget_row = self.ui.create_row_widget(properties={"size-policy-horizontal": "maximum"})

        toolbar_row_widget = self.ui.create_row_widget()
        toolbar_row_widget.add(self.__toolbar_widget_row)
        toolbar_row_widget.add_stretch()

        self.widget.add(toolbar_row_widget)

        # make a map from widget_id to widget factory.
        widget_factories: typing.Dict[str, typing.Callable[..., Declarative.HandlerLike]] = dict()
        for component in Registry.get_components_by_type("toolbar-widget"):
            widget_factories[component.toolbar_widget_id] = component

        # define the order of widgets.
        # this part is hard coded for now; needs some work to make it dynamically order widgets as they become
        # available from packages.
        widget_id_list = [
            "nion.swift.toolbar-widget.tool-mode",
            "nion.swift.toolbar-widget.raster-zoom",
            "nion.swift.toolbar-widget.workspace",
        ]

        if Feature.FeatureManager().is_feature_enabled("feature.command_palette"):
            widget_id_list.append("nion.swift.toolbar-widget.commands")

        # add the widgets.
        for widget_id in widget_id_list:
            widget_factory = widget_factories[widget_id]
            widget_handler = widget_factory(document_controller=self.document_controller)
            widget = Declarative.DeclarativeWidget(self.ui, self.document_controller.event_loop, widget_handler)
            widget_section = self.ui.create_row_widget()
            widget_section.add(widget)
            section_bar = self.ui.create_canvas_widget(properties={"width": 9, "size_policy_vertical": "expanding"})

            def draw(drawing_context: DrawingContext.DrawingContext, canvas_size: Geometry.IntSize, *args: typing.Any, **kwargs: typing.Any) -> None:
                with drawing_context.saver():
                    drawing_context.rect(0, 0, canvas_size.width, canvas_size.height)
                    drawing_context.fill_style = "#DDD"
                    drawing_context.stroke_style = "#AAA"
                    drawing_context.fill()
                    drawing_context.stroke()

            section_bar.canvas_item.add_canvas_item(CanvasItem.DrawCanvasItem(draw))
            self.__toolbar_widget_row.add(section_bar)
            self.__toolbar_widget_row.add_spacing(8)
            self.__toolbar_widget_row.add(widget_section)
            self.__toolbar_widget_row.add_spacing(8)

        end_divider = self.ui.create_canvas_widget(properties={"width": 1, "size_policy_vertical": "expanding"})
        end_divider.canvas_item.add_canvas_item(CanvasItem.DividerCanvasItem(color="#888"))
        self.__toolbar_widget_row.add(end_divider)

    def close(self) -> None:
        self.__component_registered_listener.close()
        self.__component_registered_listener = typing.cast(typing.Any, None)
        super().close()

    def __component_registered(self, component: typing.Callable[..., Declarative.HandlerLike], component_types: typing.Set[str]) -> None:
        if "toolbar-widget" in component_types:
            self.__toolbar_widget_row.add_spacing(12)
            self.__toolbar_widget_row.add(Declarative.DeclarativeWidget(self.ui, self.document_controller.event_loop, component(document_controller=self.document_controller)))


class ActionCommandHandler(Declarative.Handler):
    def __init__(self, ui: UserInterface.UserInterface, action: Profile.Action) -> None:
        super().__init__()
        self.__ui = ui
        self.action = action
        u = Declarative.DeclarativeUI()
        script_row = u.create_row(
            u.create_row(
                u.create_label(text=_("Action Id"), width=80),
                u.create_line_edit(text="@binding(action_id)", width=360)),
            u.create_stretch(), spacing=8)
        self.ui_view = u.create_column(
            u.create_row(u.create_label(text=_("Command: Generic")), u.create_stretch()),
            script_row,
            spacing=8)

    @property
    def action_id(self) -> typing.Optional[str]:
        action_id = self.action.action_id
        return str(action_id) if action_id else None

    @action_id.setter
    def action_id(self, value: typing.Optional[str]) -> None:
        if value:
            self.action.action_id = str(value)
        else:
            self.action.action_id = "application.uncommand"
        self.notify_property_changed("action_id")


class RunScriptActionCommandHandler(Declarative.Handler):
    def __init__(self, ui: UserInterface.UserInterface, action: Profile.RunScriptAction) -> None:
        super().__init__()
        self.__ui = ui
        self.action = action
        u = Declarative.DeclarativeUI()
        script_row = u.create_row(u.create_row(u.create_label(text=_("Script Path"), width=80),
                                               u.create_line_edit(text="@binding(script_path)", width=360)),
                                  u.create_push_button(text="...", on_clicked="handle_script_path", style="minimal"),
                                  u.create_stretch(), spacing=8)
        self.ui_view = u.create_column(
            u.create_row(u.create_label(text=_("Command: Run Script")), u.create_stretch()),
            script_row,
            spacing=8
        )

    @property
    def script_path(self) -> typing.Optional[str]:
        script_path = self.action.script_path
        return str(script_path) if script_path else None

    @script_path.setter
    def script_path(self, value: typing.Optional[str]) -> None:
        if value:
            self.action.script_path = pathlib.Path(value)
        else:
            self.action.script_path = None
        self.notify_property_changed("script_path")

    def handle_script_path(self, widget: UserInterface.Widget) -> None:
        ui = self.__ui
        PERSISTENT_DIRECTORY_KEY = "script_commands_dir"
        filter_str = "Scripts (*.py);;All Files (*.*)"
        import_dir = ui.get_persistent_string(PERSISTENT_DIRECTORY_KEY, ui.get_document_location())
        paths, selected_filter, selected_directory = ui.get_file_paths_dialog(_("Script Path"), import_dir, filter_str)
        if len(paths) == 1:
            ui.set_persistent_string(PERSISTENT_DIRECTORY_KEY, selected_directory)
            path = pathlib.Path(paths[0])
            self.script_path = str(path)


class ToolbarCommandDetailHandler(Declarative.Handler):
    # a master-detail for each top level item

    def __init__(self, action_command: Profile.ActionCommand, ui: UserInterface.UserInterface) -> None:
        super().__init__()
        self.action_command = action_command
        self.__ui = ui
        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_column(
            u.create_row(u.create_label(text=_("Title"), width=80),
                         u.create_line_edit(text="@binding(action_command.title)", width=100), u.create_stretch()),
            u.create_row(u.create_label(text=_("Hint"), width=80),
                         u.create_line_edit(text="@binding(action_command.tool_tip)", width=240), u.create_stretch()),
            u.create_column(items="action_command.actions", item_component_id="action", spacing=8),
            # disabled until developed further
            # u.create_row(u.create_push_button(text=_("Add Run Script"), on_clicked="add_run_script_action", style="minimal"),
            #              u.create_push_button(text=_("Add Action by Id"), on_clicked="add_action", style="minimal")
            #              ),
            u.create_stretch(),
            spacing=8,
            width=480
        )

    def add_run_script_action(self, widget: UserInterface.Widget) -> None:
        self.action_command.append_action(Profile.RunScriptAction())

    def add_action(self, widget: UserInterface.Widget) -> None:
        self.action_command.append_action(Profile.Action())

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None,
                       **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        if component_id == "action":
            if isinstance(item, Profile.RunScriptAction):
                return RunScriptActionCommandHandler(self.__ui, item)
            else:
                return ActionCommandHandler(self.__ui, typing.cast(Profile.Action, item))
        return None


class ToolbarCommandDialog(Declarative.WindowHandler):
    def __init__(self, document_controller: DocumentController.DocumentController, profile: Profile.Profile) -> None:
        super().__init__()
        self.__document_controller = document_controller
        self.__profile = profile
        self.dialog_id = "toolbar-command-dialog"
        u = Declarative.DeclarativeUI()

        action_commands = ListModel.FilteredListModel(container=ListModel.ObservedListModel(profile, "action_commands"))

        def add_item() -> None:
            action_command = Profile.ActionCommand(actions=[Profile.RunScriptAction()])
            action_command.title = _("New Command")
            action_command.tool_tip = _("New command hint.")
            self.__profile.append_action_command(action_command)

        def remove_item(item: typing.Any) -> None:
            action_command = typing.cast(Profile.ActionCommand, item)
            self.__profile.remove_action_command(action_command)

        def make_component(item: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
            return ToolbarCommandDetailHandler(item, document_controller.ui)

        list_props = {"width": 160, "min_height": 360}
        self.__md_browser = EntityBrowser.MasterDetailHandler(action_commands, "items", typing.cast(EntityBrowser.DynamicWidgetConstructorFn, make_component), "title", None, list_props, add_item_fn=add_item, remove_item_fn=remove_item)

        content = u.create_column(u.create_component_instance(identifier="content"), u.create_stretch())

        window = u.create_window(content, title=_("Toolbar Commands"), margin=12, window_style="tool")
        self.run(window, parent_window=document_controller, persistent_id=self.dialog_id)
        self.__document_controller.register_dialog(self.window)

    def close(self) -> None:
        setattr(self.__document_controller, f"_{self.dialog_id}_dialog", None)
        super().close()

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        if component_id == "content":
            return self.__md_browser
        return None


def open_toolbar_command_dialog(document_controller: DocumentController.DocumentController, profile: Profile.Profile) -> None:
    ToolbarCommandDialog(document_controller, profile)


Feature.FeatureManager().add_feature(Feature.Feature("feature.command_palette", "Toolbar command palette for running scripts (requires restart)"))
