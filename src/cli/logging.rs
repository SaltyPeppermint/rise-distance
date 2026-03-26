use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

static LOG_FILE: Mutex<Option<File>> = Mutex::new(None);

/// Initialize the global log file. Call once at the start of `main` after creating the run folder.
///
/// # Panics
/// Panics if the log file cannot be created.
pub fn init_log(run_folder: &Path) {
    let file = File::create(run_folder.join("run.log")).expect("Failed to create run.log");
    *LOG_FILE.lock().unwrap() = Some(file);
}

/// Write a formatted message to both stdout and the log file.
#[doc(hidden)]
pub fn _tee_print(args: std::fmt::Arguments<'_>) {
    print!("{args}");
    if let Some(f) = LOG_FILE.lock().unwrap().as_mut() {
        let _ = f.write_fmt(args);
    }
}

/// Like `println!`, but also writes to the run log file.
#[macro_export]
macro_rules! tee_println {
    () => {
        $crate::cli::_tee_print(format_args!("\n"))
    };
    ($($arg:tt)*) => {{
        #[allow(clippy::used_underscore_items)]
        $crate::cli::_tee_print(format_args!($($arg)*));
        #[allow(clippy::used_underscore_items)]
        $crate::cli::_tee_print(format_args!("\n"));
    }};
}
