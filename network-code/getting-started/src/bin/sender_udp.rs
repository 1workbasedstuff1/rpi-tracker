use std::net::{SocketAddr, UdpSocket};
use std::thread;

fn main() -> std::io::Result<()> {
    println!("hello world");

    let target = std::env::args().nth(1).expect("usage: sender <ip:port>");
    let sock = UdpSocket::bind("0.0.0.0")?;

    let mut counter: u64 = 0;

    loop {
        sock.send_to(&counter.to_be_bytes(), &target).unwrap();
        counter += 1;
    }
}
