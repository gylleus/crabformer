use std::{io, sync::Arc, time::Duration};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use parking_lot::Mutex;
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, List, ListItem, Paragraph},
};

use crate::metrics::TrainingMetricsHandle;

const CRAB_ASCII: &str = r#"
           ██████░          ██████
        ▓██▓     ████   ░███░    ████
       ██░    ███           ░██▓   ░██
     ██▓█▒████                  ████████
    ▒█ ▓██                         ███ ██
    ██▓█         ██▒   🦀  ██         ██▓█
    ▒█░▒██        ██     ██        ██▒ ██
 ██▒  ███▓██  █████████████████░ ██▓▓██   ██
 ▒███░   ██████▒░░▒░     ░▓░ ░▓█████    ████
   ▓███░   █▒░   ░▒▒░    ░▓░   ░░▓░   ████
      ██████░░  ░░▒▒░ ░  ▒▓░   ░░▓█████▒
          ░▓░░   ░░▒░    ▓▒░░  ░▒▓█
 ████████████▓▒▒░░▒▓▓░ ░▒█▒░░░▒▒████████████
            ██▓▒▒▒▒▒▓▓▓▓▓▒░░░▒▒▓█
     █████████████▒▒▒▒▒▒▒▒▒▒████████████
   █████        ░██▓▓▒▒▒▒▓███        █████
  ████    ▒█████▓███████████▓▓█████     ███▒
  ▒     ░████▒▒▒             ░░░▓████
       ███                         ░██
      ███                           ▓██
       █                             █░
"#;

pub struct Dashboard {
    metrics_handle: TrainingMetricsHandle,
    should_quit: Arc<Mutex<bool>>,
    crab_animation: CrabAnimation,
}

pub struct CrabAnimation {
    position: i32,
    move_speed: i32,
    going_left: bool,
    half_width: u16,
}

impl CrabAnimation {
    fn is_oob(&self, terminal_width: u16) -> bool {
        self.position <= 0_i32 || self.position + self.half_width as i32 > terminal_width as i32
    }
}

impl Dashboard {
    pub fn new(metrics_handle: TrainingMetricsHandle) -> Self {
        Self {
            metrics_handle,
            should_quit: Arc::new(Mutex::new(false)),
            crab_animation: CrabAnimation {
                position: 64,
                move_speed: 2,
                going_left: false,
                half_width: 50,
            },
        }
    }

    pub fn run(&mut self) -> Result<(), io::Error> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Run dashboard app
        let result = self.run_app(&mut terminal);

        // Restore terminal
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        result
    }

    fn run_app(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    ) -> Result<(), io::Error> {
        loop {
            terminal.draw(|f| {
                // Create layout
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3),      // Title
                        Constraint::Length(3),      // Current metrics
                        Constraint::Percentage(50), // Loss chart
                        Constraint::Min(5),         // Layer details
                        Constraint::Length(3),      // Progress bar
                        Constraint::Length(20),     // Crab animation
                    ])
                    .split(f.area());

                // Title
                self.render_title(f, chunks[0]);

                // Render all metric tabs
                {
                    let metrics = self.metrics_handle.lock();
                    // Current metrics
                    self.render_current_metrics(f, chunks[1], &metrics);

                    // Loss chart
                    self.render_loss_chart(f, chunks[2], &metrics);

                    // Layer details
                    self.render_layer_details(f, chunks[3], &metrics);

                    // Progress bar
                    self.render_progress_bar(f, chunks[4], &metrics);
                }

                // Crab animation
                self.render_crab(f, chunks[5]);
            })?;

            // Check for quit signal
            if *self.should_quit.lock() {
                break;
            }

            // Poll for events with timeout
            if event::poll(Duration::from_millis(100))?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
                && key.code == KeyCode::Char('q')
            {
                *self.should_quit.lock() = true;
                break;
            }
        }

        Ok(())
    }

    fn render_title(&self, f: &mut Frame, area: Rect) {
        let title = Paragraph::new("Crabformer Training Dashboard | (Press 'q' to Quit)")
            .style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .block(Block::default().borders(Borders::ALL));
        f.render_widget(title, area);
    }

    fn render_current_metrics(
        &self,
        f: &mut Frame,
        area: Rect,
        metrics: &parking_lot::MutexGuard<crate::metrics::TrainingMetrics>,
    ) {
        let elapsed = metrics.start_time.elapsed();
        let elapsed_secs = elapsed.as_secs();
        let hours = elapsed_secs / 3600;
        let minutes = (elapsed_secs % 3600) / 60;
        let seconds = elapsed_secs % 60;

        let batches_per_sec = if elapsed.as_secs() > 0 {
            metrics.processed_batches as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        let text = format!(
            "Loss: {:.6} | Batches: {} | Time: {:02}:{:02}:{:02} | Speed: {:.2} batches/sec",
            metrics.current_loss(),
            metrics.processed_batches,
            hours,
            minutes,
            seconds,
            batches_per_sec
        );

        let paragraph = Paragraph::new(text)
            .style(Style::default().fg(Color::Green))
            .block(Block::default().borders(Borders::ALL).title("Metrics"));

        f.render_widget(paragraph, area);
    }

    fn render_loss_chart(
        &self,
        f: &mut Frame,
        area: Rect,
        metrics: &parking_lot::MutexGuard<crate::metrics::TrainingMetrics>,
    ) {
        let loss_history = &metrics.loss_history;

        if loss_history.is_empty() {
            let placeholder = Paragraph::new("Loss chart will appear once training begins...")
                .style(Style::default().fg(Color::Gray))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Loss Over Time"),
                );
            f.render_widget(placeholder, area);
            return;
        }

        let max_loss = loss_history
            .iter()
            .map(|(_, loss)| *loss)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_loss = loss_history
            .iter()
            .map(|(_, loss)| *loss)
            .fold(f32::INFINITY, f32::min);

        // Calculate target number of points based on chart width
        // Subtract borders (2) and some padding, estimate ~2 chars per point
        let chart_width = area.width.saturating_sub(4) as usize;
        let target_points = (chart_width / 2).clamp(50, 200);

        // Sample or extend data to match target_points
        let data: Vec<(f64, f64)> = if loss_history.len() <= target_points {
            // Few samples: extend to full width by repeating last value
            let max_batch = loss_history.last().map(|(b, _)| *b as f64).unwrap_or(1.0);
            let last_loss = loss_history.last().map(|(_, l)| *l as f64).unwrap_or(0.0);

            // First, add all existing data points
            let mut points: Vec<(f64, f64)> = loss_history
                .iter()
                .map(|(batch, loss)| (*batch as f64, *loss as f64))
                .collect();

            // Then extend to target_points by filling with the last value
            let batch_step = max_batch / (target_points - 1) as f64;
            for i in loss_history.len()..target_points {
                let batch = max_batch.max(i as f64 * batch_step);
                points.push((batch, last_loss));
            }

            points
        } else {
            // Many samples: downsample using uniform sampling
            let step = loss_history.len() as f64 / target_points as f64;
            let mut points = Vec::with_capacity(target_points);

            for i in 0..target_points {
                let idx = (i as f64 * step).floor() as usize;
                let idx = idx.min(loss_history.len() - 1);
                let (batch, loss) = loss_history[idx];
                points.push((batch as f64, loss as f64));
            }

            // Always include the last point
            if let Some(&(batch, loss)) = loss_history.last()
                && points.last().map(|&(b, _)| b) != Some(batch as f64)
            {
                points.push((batch as f64, loss as f64));
            }

            points
        };

        let dataset = Dataset::default()
            .name("Loss")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Yellow))
            .data(&data);

        let max_batch = data.last().map(|(b, _)| *b).unwrap_or(1.0);

        // Create X-axis labels
        let x_labels = [
            "0".to_string(),
            format!("{:.0}", max_batch / 2.0),
            format!("{:.0}", max_batch),
        ];
        let x_labels_str: Vec<&str> = x_labels.iter().map(|s| s.as_str()).collect();

        // Create Y-axis labels
        let y_min = (min_loss * 0.9) as f64;
        let y_max = (max_loss * 1.1) as f64;
        let y_labels = [
            format!("{:.4}", y_min),
            format!("{:.4}", (y_min + y_max) / 2.0),
            format!("{:.4}", y_max),
        ];
        let y_labels_str: Vec<&str> = y_labels.iter().map(|s| s.as_str()).collect();

        let chart = Chart::new(vec![dataset])
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Loss Over Time"),
            )
            .x_axis(
                Axis::default()
                    .title("Batch")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, max_batch])
                    .labels(x_labels_str),
            )
            .y_axis(
                Axis::default()
                    .title("Loss")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([y_min, y_max])
                    .labels(y_labels_str),
            );

        f.render_widget(chart, area);
    }

    fn render_progress_bar(
        &self,
        f: &mut Frame,
        area: Rect,
        metrics: &parking_lot::MutexGuard<crate::metrics::TrainingMetrics>,
    ) {
        // Split the area into two columns for the two progress bars
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Calculate batches in current epoch
        let batches_in_epoch = if metrics.batches_per_epoch > 0 {
            metrics.processed_batches % metrics.batches_per_epoch
        } else {
            0
        };

        // Epoch progress
        let epoch_percentage = if metrics.epochs > 0 {
            ((metrics.current_epoch as f64 / metrics.epochs as f64) * 100.0).min(100.0) as u16
        } else {
            0
        };

        let epoch_label = format!(
            "Epoch {}/{} ({}%)",
            metrics.current_epoch, metrics.epochs, epoch_percentage
        );

        let epoch_gauge = Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Epoch Progress"),
            )
            .gauge_style(Style::default().fg(Color::Green).bg(Color::Black))
            .percent(epoch_percentage)
            .label(epoch_label);

        // Batch progress within current epoch
        let batch_percentage = if metrics.batches_per_epoch > 0 {
            ((batches_in_epoch as f64 / metrics.batches_per_epoch as f64) * 100.0) as u16
        } else {
            0
        };

        let batch_label = format!(
            "Batch {}/{} ({}%)",
            batches_in_epoch, metrics.batches_per_epoch, batch_percentage
        );

        let batch_gauge = Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Batch Progress"),
            )
            .gauge_style(Style::default().fg(Color::Yellow).bg(Color::Black))
            .percent(batch_percentage)
            .label(batch_label);

        f.render_widget(epoch_gauge, chunks[0]);
        f.render_widget(batch_gauge, chunks[1]);
    }

    fn render_crab(&mut self, f: &mut Frame, area: Rect) {
        // Update crab position before rendering
        self.crab_animation.position += if self.crab_animation.going_left {
            -self.crab_animation.move_speed
        } else {
            self.crab_animation.move_speed
        };

        if self.crab_animation.is_oob(area.width) {
            // Reverse direction
            self.crab_animation.going_left = !self.crab_animation.going_left;
        }

        // Calculate padding to position the crab horizontally
        let pos = self.crab_animation.position as usize;

        // Add padding spaces to move the crab
        let crab_lines: Vec<String> = CRAB_ASCII
            .lines()
            .map(|line| format!("{:pos$}{}", "", line, pos = pos))
            .collect();

        let crab_text = crab_lines.join("\n");

        let crab_widget = Paragraph::new(crab_text)
            .style(Style::default().fg(Color::Red))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Cooking the crabformer..."),
            );

        f.render_widget(crab_widget, area);
    }

    fn render_layer_details(
        &self,
        f: &mut Frame,
        area: Rect,
        metrics: &parking_lot::MutexGuard<crate::metrics::TrainingMetrics>,
    ) {
        let total_transformer_blocks = metrics.total_transformer_blocks.max(1) as f64;

        // Helper macro to format timing values
        macro_rules! ms {
            ($duration:expr) => {
                $duration.as_secs_f64() * 1000.0
            };
        }

        // Helper macro to create rows with consistent formatting
        macro_rules! row {
            ($label:expr, $fwd:expr, $bwd:expr) => {
                ListItem::new(format!(
                    "{:<23}Fwd: {:>6.2}ms  Bwd: {:>6.2}ms",
                    $label, $fwd, $bwd
                ))
            };
            ($label:expr, $fwd:expr, $bwd:expr, $total:expr) => {
                ListItem::new(format!(
                    "{:<23}Fwd: {:>6.2}ms  Bwd: {:>6.2}ms - (Sum blocks: {:>6.2}ms)",
                    $label, $fwd, $bwd, $total
                ))
            };
        }

        let items: Vec<ListItem> = vec![
            row!(
                "Transformer block:",
                ms!(metrics.transformer_block_duration.avg_forward_duration()),
                ms!(metrics.transformer_block_duration.avg_backward_duration()),
                (ms!(metrics.transformer_block_duration.avg_forward_duration())
                    + ms!(metrics.transformer_block_duration.avg_backward_duration()))
                    * total_transformer_blocks
            ),
            row!(
                "Attention:",
                ms!(metrics.attention_duration.avg_forward_duration()),
                ms!(metrics.attention_duration.avg_backward_duration()),
                (ms!(metrics.attention_duration.avg_forward_duration())
                    + ms!(metrics.attention_duration.avg_backward_duration()))
                    * total_transformer_blocks
            ),
            row!(
                "Feed Forward:",
                ms!(metrics.feed_forward_duration.avg_forward_duration()),
                ms!(metrics.feed_forward_duration.avg_backward_duration()),
                (ms!(metrics.feed_forward_duration.avg_forward_duration())
                    + ms!(metrics.feed_forward_duration.avg_backward_duration()))
                    * total_transformer_blocks
            ),
            row!(
                "Layer Norm:",
                ms!(metrics.layer_norm_duration.avg_forward_duration()),
                ms!(metrics.layer_norm_duration.avg_backward_duration())
            ),
            row!(
                "Token Embedding:",
                ms!(metrics.token_embedding_duration.avg_forward_duration()),
                ms!(metrics.token_embedding_duration.avg_backward_duration())
            ),
            row!(
                "Pos Embedding:",
                ms!(metrics.positional_embedding_duration.avg_forward_duration()),
                ms!(metrics
                    .positional_embedding_duration
                    .avg_backward_duration())
            ),
            row!(
                "Output Layer:",
                ms!(metrics.output_layer_duration.avg_forward_duration()),
                ms!(metrics.output_layer_duration.avg_backward_duration())
            ),
        ];

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Compute Durations per Layer"),
            )
            .style(Style::default().fg(Color::White));

        f.render_widget(list, area);
    }
}

/// Creates and starts a training metrics dashboard in new thread.
/// Returns a handle to signal the dashboard to quit, and the thread handle.
pub fn start_dashboard(
    metrics_handle: TrainingMetricsHandle,
) -> (Arc<Mutex<bool>>, std::thread::JoinHandle<()>) {
    let mut dashboard = Dashboard::new(metrics_handle);
    let should_quit_handle = dashboard.should_quit.clone();

    let handle = std::thread::spawn(move || {
        if let Err(e) = dashboard.run() {
            eprintln!("Dashboard error: {}", e);
        }
    });
    (should_quit_handle, handle)
}
