# BedFrame cookbook

Recipes for the declarative layer (`dream::ui::view`). Geometry comes from the layout solver.

## Basics

```rust,ignore
ui.layout_root(area, Style::column().pad(8.0).gap(4.0), |ui| {
    ui.add(Text::new("Title").role(TypeRole::Heading));
    if ui.add(Button::new("Done")).clicked { /* … */ }
});
```

- The build closure is `FnMut` and may run twice per frame. It must be idempotent: record state from events (`if resp.clicked { count += 1 }`), never assign from responses unconditionally.
- `ui.rect_of("key")` returns the solved rect after `layout_root` returns.
- `ui.draw` and `DrawList` work inside `slot` closures.

## Toolbar

```rust,ignore
ui.layout_root(area, Style::row().gap(4.0).align(Align::Center), |ui| {
    ui.add(IconButton::new("##back", Icon::ChevronLeft));
    ui.add_styled(Text::new("Scene").role(TypeRole::Heading), Style::default().grow(1.0));
    ui.add(Button::new("Play").variant(ButtonVariant::Primary));
});
```

## Form rows

There is no `Form` widget yet. Use a column of two-column rows:

```rust,ignore
ui.layout_root(area, Style::column().pad(12.0).gap(8.0), |ui| {
    for (label, value) in fields.iter_mut() {
        ui.row(Style::default().justify(Justify::SpaceBetween).align(Align::Center), |ui| {
            ui.add(Text::new(label));
            ui.add_styled(TextField::new(&format!("##{label}"), value),
                Style::default().w_px(180.0));
        });
    }
});
```

## Cards

There is no `Grid` widget yet. Use fixed-height rows:

```rust,ignore
ui.layout_root(area, Style::column().gap(8.0), |ui| {
    for row in cards.chunks_mut(2) {
        ui.row(Style::default().gap(8.0).fill_w(), |ui| {
            for card in row {
                ui.container(
                    Style::column().grow(1.0).h_px(120.0).pad(8.0).gap(4.0),
                    Paint::surface(theme),
                    |ui| {
                        ui.add(Text::new(&card.title).role(TypeRole::Body));
                        ui.add(Text::new(&card.sub).role(TypeRole::Caption));
                    },
                );
            }
        });
    }
});
```

## Scrollable page

Reference: `src/app/settings_page.rs`.

```rust,ignore
ui.layout_root(page, Style::column().fill(), |ui| {
    ui.container(Style::column().fill().scroll_y(), Paint::none(), |ui| {
        ui.column(Style::column().pad(Space::S6).gap(Space::S5).max_w(440.0), |ui| {
            ui.add(Text::new("Settings").role(TypeRole::Title));
        });
    });
});
```

## Scroll feel

Every scroll view shares one physics model. Tune it globally on `ScrollConfig`:

```rust,ignore
ui.scroll_config = ScrollConfig {
    line_height: 40.0,     // pixels per wheel line
    smoothness: 0.06,      // glide time constant; smaller is snappier
    page_fraction: 0.85,   // PageUp/PageDown and track clicks
    keyboard: true,        // arrows and page keys scroll the hovered view
    chain: true,           // wheel overflow passes to the parent view
    ..ScrollConfig::default()
};

ui.scroll_config = ScrollConfig::no_momentum(); // notches jump, nothing coasts
```

Override per view by saving and restoring around the build:

```rust,ignore
let saved = ui.scroll_config;
ui.scroll_config = ScrollConfig::no_momentum();
let body = ui.scroll_begin("inspector", area, content_h);
// …rows at `body`…
ui.scroll_end();
ui.scroll_config = saved;
```

Programmatic scrolling. Calls may come before or after the view builds:

```rust,ignore
ui.scroll_to("log", ScrollToOptions::smooth(f32::INFINITY)); // pin to end
ui.scroll_by("log", -48.0, ScrollBehavior::Auto);
ui.scroll_into_view("files", Rect::new(0.0, row_y, width, row_h),
    ScrollIntoViewOptions { block: ScrollAlign::Nearest, behavior: ScrollBehavior::Smooth });
```

Read state with `scroll_state(label)` (`offset`, `content`) or `scroll_metrics(label)` (adds `viewport`, `max_offset`, `velocity`, `animating`, `progress()`).

## HUD overlay

`stack` places children by offset, so overlays do not fight the layout:

```rust,ignore
ui.layout_root(screen, Style::column().fill(), |ui| {
    ui.stack(Style::default().fill(), |ui| {
        ui.add_styled(Text::new("60 fps").role(TypeRole::Caption),
            Style::default().place(AlignX::Start, AlignY::Start));
        ui.add_styled(IconView::new("##aim", Icon::Plus),
            Style::default().place(AlignX::Center, AlignY::Center));
    });
});
```

## State and context

`use_state` returns a shared handle. Its key follows `push_id` scopes, and the initializer runs once even when layout replays:

```rust,ignore
let count = ui.use_state("counter", || 0usize);
ui.with_context(&count, |ui| {
    ui.layout_root(area, Style::column().gap(8.0), |ui| {
        let count = ui.use_context::<usize>().expect("counter provider");
        ui.add(Text::new(&format!("Count: {}", count.get())));
        if ui.add(Button::new("Increment")).clicked {
            count.update(|n| *n += 1);
        }
    });
});
```

- The nearest provider of a type wins. Use newtypes to separate contexts that share a primitive type.
- `State::new(value)` creates a host-owned provider.
- State changed outside a UI frame must schedule a frame.

## Presence

Keep a widget mounted while it animates out:

```rust,ignore
let p = ui.presence("notice", visible, 0.18);
if p.is_present() {
    let color = theme.fg.with_alpha((255.0 * p.opacity).round() as u8);
    ui.label(notice_rect, "Saved", theme.label(font).color(color));
}
```

- Call `presence` every frame for the same key.
- During exit, `is_interactive()` is false. Disable the widget's controls then.
- Reduced motion settles immediately.
