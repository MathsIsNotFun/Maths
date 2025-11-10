import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.path import Path
import matplotlib.patches as patches
from collections import defaultdict
import subprocess
import platform

# Try to import sound library
try:
    if platform.system() == 'Darwin':  # macOS
        import os
        SOUND_AVAILABLE = True
        def play_knock():
            # Use a more woody sound
            os.system('afplay /System/Library/Sounds/Pop.aiff > /dev/null 2>&1 &')
    elif platform.system() == 'Windows':
        import winsound
        SOUND_AVAILABLE = True
        def play_knock():
            # Lower frequency for more woody sound
            winsound.Beep(400, 30)  # 400Hz for 30ms - deeper, shorter knock
    else:  # Linux
        SOUND_AVAILABLE = False
        def play_knock():
            pass
except:
    SOUND_AVAILABLE = False
    def play_knock():
        pass

# Function to generate audio file
def generate_audio_track(frames, output_file='hasse_audio.wav', base_interval=400):
    """
    Generate a WAV file with knock sounds timed to match the animation
    
    Parameters:
    - frames: list of frame data dictionaries
    - output_file: name of output WAV file
    - base_interval: milliseconds per frame
    """
    try:
        from scipy.io import wavfile
        
        print(f"\nGenerating audio track: {output_file}")
        
        # Audio parameters
        sample_rate = 44100  # CD quality
        knock_duration = 0.03  # 30ms
        knock_freq = 400  # Hz - woody knock sound
        
        # Calculate total duration
        total_duration = len(frames) * (base_interval / 1000.0)  # in seconds
        total_samples = int(total_duration * sample_rate)
        
        # Create silent audio array
        audio = np.zeros(total_samples)
        
        # Add knock sound for each frame where a node appears
        current_time = 0
        for i, frame_data in enumerate(frames):
            if frame_data.get('play_sound', False):
                # Generate knock sound (sine wave with envelope)
                knock_samples = int(knock_duration * sample_rate)
                t = np.linspace(0, knock_duration, knock_samples)
                
                # Sine wave
                knock = np.sin(2 * np.pi * knock_freq * t)
                
                # Apply envelope (fade in/out) for more natural sound
                envelope = np.exp(-t * 50)  # Exponential decay
                knock = knock * envelope
                
                # Normalize
                knock = knock * 0.3  # Keep it quiet
                
                # Insert into audio at correct position
                start_sample = int(current_time * sample_rate)
                end_sample = start_sample + knock_samples
                
                if end_sample < total_samples:
                    audio[start_sample:end_sample] += knock
            
            current_time += base_interval / 1000.0
        
        # Normalize audio to prevent clipping
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Convert to 16-bit PCM
        audio_int16 = np.int16(audio * 32767)
        
        # Save as WAV
        wavfile.write(output_file, sample_rate, audio_int16)
        
        print(f"✓ Audio saved: {output_file}")
        print(f"  Duration: {total_duration:.1f} seconds")
        print(f"  Knocks: {sum(1 for f in frames if f.get('play_sound', False))}")
        print(f"\nTo combine with video:")
        print(f"  ffmpeg -i hasse_tree.mp4 -i {output_file} -c:v copy -c:a aac -shortest hasse_tree_with_audio.mp4")
        
        return True
        
    except ImportError:
        print("\n⚠️  scipy not available - cannot generate audio file")
        print("Install with: pip install scipy")
        return False
    except Exception as e:
        print(f"\n⚠️  Error generating audio: {e}")
        return False

print("Note: Sound effects don't export to video files (MP4/GIF)")
print("Sound only plays during live animation viewing")
print("But we can generate a separate audio file to combine later!")

# ============================================================================
# Helper functions
# ============================================================================

def get_all_numbers_up_to(max_n):
    """Get all integers from 1 to max_n"""
    return list(range(1, max_n + 1))

def count_divisors(n):
    """Count the number of divisors of n"""
    count = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count

def get_divisors_in_set(n, elements_set):
    """Get divisors of n that are in elements_set"""
    return [d for d in elements_set if n % d == 0]

# Track when each node was last used (for leaf color aging)
node_last_used = {}

def get_leaf_color(elem, current_n, is_highlighted, frames_since_used):
    """
    Get color like a tree leaf - aging effect
    Green (just used) → Yellow (aging) → Red (old) → Brown (very old)
    When used again, returns to green!
    """
    if is_highlighted:
        return '#2ecc71'  # Emerald green
    
    if frames_since_used <= 5:
        return '#7bed9f'  # Light green
    elif frames_since_used <= 15:
        return '#ffd32a'  # Yellow
    elif frames_since_used <= 30:
        return '#ff793f'  # Orange
    elif frames_since_used <= 50:
        return '#e74c3c'  # Red
    else:
        return '#8b4513'  # Saddle brown

# ============================================================================
# Build unified Hasse diagram
# ============================================================================

def build_unified_hasse_edges(elements):
    """Build edges for unified Hasse diagram"""
    edges = []
    elements_sorted = sorted(elements)
    
    for a in elements_sorted:
        for b in elements_sorted:
            if b < a and a % b == 0:
                is_cover = True
                for c in elements_sorted:
                    if b < c < a and a % c == 0 and c % b == 0:
                        is_cover = False
                        break
                if is_cover:
                    edges.append((b, a))
    
    return edges

def compute_levels_unified(elements, edges):
    """Assign vertical levels based on divisibility chains"""
    levels = {1: 0}
    
    children = defaultdict(list)
    for parent, child in edges:
        children[parent].append(child)
    
    queue = [1]
    visited = {1}
    
    while queue:
        current = queue.pop(0)
        for child in children[current]:
            if child not in visited:
                levels[child] = levels[current] + 1
                visited.add(child)
                queue.append(child)
    
    return levels

def layout_organic_tree(elements, edges):
    """
    Compute organic tree layout - grows UPWARD from root
    Node 1 STAYS at (0,0), other nodes move outward as tree grows
    """
    levels = compute_levels_unified(elements, edges)
    
    # Build parent-child relationships
    children = defaultdict(list)
    for parent, child in edges:
        children[parent].append(child)
    
    positions = {}
    
    max_n = max(elements)
    
    # Much more vertical spacing
    vertical_spacing = 25
    
    # Horizontal spacing based on VALUE
   # horizontal_scale = 0.15 if max_n > 500 else (0.3 if max_n > 200 else 0.5)
    horizontal_scale = 120/max_n
    
    # ROOT ALWAYS AT ORIGIN
    positions[1] = (0, 0)
    
    # For other nodes, position based on value relative to middle
    for elem in elements:
        if elem == 1:
            continue
        level = levels.get(elem, 0)
        y = level * vertical_spacing
        
        # x position based on value, but centered around middle of current range
        # This keeps things balanced
        x = (elem - max_n/2) * horizontal_scale
        positions[elem] = (x, y)
    
    # DON'T recenter - keep node 1 at origin!
    
    return positions, levels

def draw_curved_edge(ax, x1, y1, x2, y2, alpha=0.3, color='gray', width=1.0, growth_factor=1.0):
    """Draw a curved edge like a tree branch - curving DOWNWARD with smooth growth"""
    
    # Simple downward curve
    curve_amount = abs(x2 - x1) * 0.3
    
    # Control point below the line
    ctrl_x = (x1 + x2) / 2
    ctrl_y = (y1 + y2) / 2 - curve_amount
    
    # Calculate the actual end point based on growth_factor
    if growth_factor < 1.0:
        # Growing - use quadratic bezier interpolation
        t = growth_factor
        
        # Bezier curve: P(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
        end_x = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
        end_y = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
        
        # Also interpolate control point for smoother growth
        growing_ctrl_x = x1 + (ctrl_x - x1) * growth_factor
        growing_ctrl_y = y1 + (ctrl_y - y1) * growth_factor
        
        verts = [(x1, y1), (growing_ctrl_x, growing_ctrl_y), (end_x, end_y)]
    else:
        # Full curve
        verts = [(x1, y1), (ctrl_x, ctrl_y), (x2, y2)]
    
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color, 
                              linewidth=width, alpha=alpha)
    ax.add_patch(patch)

def draw_organic_frame(ax, current_n, elements_so_far, edges_so_far, levels, 
                       show_nodes_for, frame_progress=1.0, frame_number=0, max_number=300, 
                       play_sound=False, is_fast=False):
    """Draw one frame with organic growth and leaf aging"""
    global node_last_used
    
    # Play knock sound when new node appears
    if play_sound and current_n and SOUND_AVAILABLE:
        play_knock()
    
    # RECALCULATE positions dynamically based on current elements!
    positions, temp_levels = layout_organic_tree(elements_so_far, edges_so_far)
    
    for elem in show_nodes_for:
        node_last_used[elem] = frame_number
    
    ax.clear()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#fffacd')
    
    max_level = max(levels.values()) if levels else 1
    
    # Draw edges
    for parent, child in edges_so_far:
        if parent in positions and child in positions:
            x1, y1 = positions[parent]
            x2, y2 = positions[child]
            
            is_active = current_n and (parent == current_n or child == current_n)
            is_divisor_edge = current_n and (parent in show_nodes_for and child in show_nodes_for)
            
            if is_active:
                draw_curved_edge(ax, x1, y1, x2, y2, alpha=0.9, 
                               color='#27ae60', width=3.0, growth_factor=frame_progress)
            elif is_divisor_edge:
                draw_curved_edge(ax, x1, y1, x2, y2, alpha=0.8, 
                               color='#52c67a', width=3.0, growth_factor=1.0)
            else:
                draw_curved_edge(ax, x1, y1, x2, y2, alpha=0.4, 
                               color='#8b4513', width=0.5, growth_factor=1.0)
    
    # Draw nodes
    for elem in elements_so_far:
        if elem not in positions:
            continue
        
        x, y = positions[elem]
        
        is_highlighted = elem in show_nodes_for
        is_current = elem == current_n
        
        if elem in node_last_used:
            frames_since = frame_number - node_last_used[elem]
        else:
            frames_since = 999
        
        color = get_leaf_color(elem, current_n, is_highlighted, frames_since)
        
        if is_current:
            size = 40 * frame_progress
            edge_width = 3
            edge_color = '#196f3d'
            glow_size = 50 * frame_progress if not is_fast else 0  # No glow when fast
            alpha = frame_progress
        elif is_highlighted:
            size = 28
            edge_width = 2
            edge_color = '#27ae60'
            glow_size = 38 if not is_fast else 0  # No glow when fast
            alpha = 1.0
        else:
            base_size = 20 if max_number <= 100 else (14 if max_number <= 500 else 10)
            size = base_size
            edge_width = 1.5
            edge_color = '#5c3317'
            glow_size = 0
            alpha = 0.7
        
        if glow_size > 0:
            ax.plot(x, y, 'o', markersize=glow_size, color=color, 
                   alpha=0.3 * alpha, zorder=2)
        
        ax.plot(x, y, 'o', markersize=size, color=color, alpha=alpha,
               markeredgecolor=edge_color, markeredgewidth=edge_width, zorder=3)
        
        if max_number <= 50:
            show_label = is_highlighted or elem == 1 or levels.get(elem, 0) == 1
        elif max_number <= 200:
            show_label = is_highlighted or elem == 1
        else:
            show_label = is_highlighted
            
        if show_label:
            fontsize = 13 if is_current else (10 if max_number <= 100 else 8)
            fontweight = 'bold' if is_highlighted else 'normal'
            label_alpha = 1.0 if is_highlighted else 0.8
            if is_highlighted or frames_since < 15:
                label_color = '#1e4620'
            else:
                label_color = 'white'
            ax.text(x, y, str(elem), ha='center', va='center', 
                   fontsize=fontsize, fontweight=fontweight, 
                   color=label_color, alpha=label_alpha, zorder=4)
    
    # Static title at top - BLACK text, fixed position, more space between lines
    title_main = 'Divisibility Tree'
    
    # Main title - always in same place
    ax.text(0.5, 0.97, title_main, transform=ax.transAxes,
           ha='center', va='top', fontsize=18, fontweight='bold',
           color='black')
    
    # Info text 
    if current_n:
        info_line = f'Adding: {current_n}    Level: {levels.get(current_n, 0)}    Divisors: {len(get_divisors_in_set(current_n, elements_so_far))}'
        ax.text(0.5, 0.92, info_line, transform=ax.transAxes,
                ha='center', va='top', fontsize=18,
                color='black')
    
    # Set limits - FIXED for entire animation so node 1 doesn't move!
    # Fixed limits for TikTok 9:16 aspect ratio
    max_x_extent = 70
#    max_y_extent = max(levels.values()) * 25.0 if levels else 200
    max_y_extent=244   
    ax.set_xlim(-max_x_extent, max_x_extent)
    ax.set_ylim(-5, max_y_extent)
    

# ============================================================================
# Animation setup
# ============================================================================

print("="*70)
print("ORGANIC GROWING HASSE DIAGRAM")
print("="*70)
print("Creating tree-like growth animation...")
print("="*70)

max_number = 512

# Define ALL speed variables ONCE - no duplicates!
# Target: 40 seconds for 300 numbers
slow_until = 12       # First 10: VERY slow
medium_until = 18     # Next 15: medium
fast_until = 990       # Next 35: fast  
# Remaining 240: VERY fast

#slow_frames = [0.17, 0.33, 0.5, 0.67, 0.83, 1.0]  # 6 frames - smooth growth
slow_frames=list(range(0,100))
slow_frames=[x /100 for x in slow_frames]

medium_frames=list(range(0,20))
medium_frames=[x /20 for x in medium_frames]


# these don't seem to do anything
slow_interval = 20000
medium_interval = 20000
fast_interval = 1
very_fast_interval = 20000

print(f"max_number = {max_number}")
print(f"Speed tiers: slow_until={slow_until}, medium_until={medium_until}, fast_until={fast_until}")
print(f"This means: 1-{slow_until} slow, {slow_until+1}-{medium_until} medium, {medium_until+1}-{fast_until} fast, {fast_until+1}-{max_number} very fast")

all_elements = get_all_numbers_up_to(max_number)
all_edges = build_unified_hasse_edges(all_elements)
# DON'T call layout here - we'll call it per frame!
all_levels = compute_levels_unified(all_elements, all_edges)

print(f"Total: {len(all_elements)} numbers, {len(all_edges)} edges")


fig, ax = plt.subplots(figsize=(14, 27), facecolor='#fffacd')
ax.set_facecolor('#fffacd')

print("\nGenerating organic growth frames with variable speed...")

frames = []




# Slow start
for n in range(1, min(slow_until + 1, max_number + 1)):
    elements_so_far = list(range(1, n + 1))
    edges_so_far = [(p, c) for p, c in all_edges if p <= n and c <= n]
    
    divisors = get_divisors_in_set(n, elements_so_far)
    show_nodes = set(divisors)
    
    for progress in slow_frames:
        frames.append({
            'current_n': n if progress >= 0.8 else n,
            'elements': elements_so_far,
            'edges': edges_so_far,
            'show_nodes': show_nodes if progress >= 0.8 else set(divisors),
            'progress': progress,
            'interval': slow_interval,
            'frame_number': len(frames),
            'play_sound': progress >= 0.8  # Sound when leaf appears
        })

# Medium speed
for n in range(slow_until + 1, min(medium_until + 1, max_number + 1)):
    elements_so_far = list(range(1, n + 1))
    edges_so_far = [(p, c) for p, c in all_edges if p <= n and c <= n]
    
    divisors = get_divisors_in_set(n, elements_so_far)
    show_nodes = set(divisors)
    
    for progress in medium_frames:
        frames.append({
            'current_n': n if progress >= 0.7 else n,
            'elements': elements_so_far,
            'edges': edges_so_far,
            'show_nodes': show_nodes if progress >= 0.7 else set(divisors),
            'progress': progress,
            'interval': medium_interval,
            'frame_number': len(frames),
            'play_sound': progress >= 0.7
        })

# Fast 
for n in range(medium_until + 1, min(fast_until + 1, max_number + 1)):
    elements_so_far = list(range(1, n + 1))
    edges_so_far = [(p, c) for p, c in all_edges if p <= n and c <= n]
    
    divisors = get_divisors_in_set(n, elements_so_far)
    show_nodes = set(divisors)
    
    # Branch frame
    frames.append({
        'current_n': n,
        'elements': elements_so_far,
        'edges': edges_so_far,
        'show_nodes': set(divisors),
        'progress': 0.5,
        'interval': fast_interval // 2,
        'frame_number': len(frames),
        'play_sound': False
    })
    
    # Leaf frame with sound
    frames.append({
        'current_n': n,
        'elements': elements_so_far,
        'edges': edges_so_far,
        'show_nodes': show_nodes,
        'progress': 1.0,
        'interval': fast_interval // 2,
        'frame_number': len(frames),
        'play_sound': False  # Knock sound!
    })

# VERY fast ending (101-300) - just 1 frame per number!
for n in range(fast_until + 1, max_number + 1):
    elements_so_far = list(range(1, n + 1))
    edges_so_far = [(p, c) for p, c in all_edges if p <= n and c <= n]
    
    divisors = get_divisors_in_set(n, elements_so_far)
    show_nodes = set(divisors + [n])
    
    # Single frame - blazing fast!
    frames.append({
        'current_n': n,
        'elements': elements_so_far,
        'edges': edges_so_far,
        'show_nodes': show_nodes,
        'progress': 0.01,
        'interval': very_fast_interval,
        'frame_number': len(frames),
        'play_sound': True
    })

final_elements = list(range(1, max_number + 1))
final_edges = [(p, c) for p, c in all_edges if p <= max_number and c <= max_number]

for i in range(360):
    frames.append({
        'current_n': None,
        'elements': final_elements,
        'edges': final_edges,
        'show_nodes': set(),
        'progress': 1.0,
        'frame_number': len(frames),
        'play_sound': False,
        'is_fast': True
    })
    
    
frames.append({
    'current_n': None,
    'elements': all_elements,
    'edges': all_edges,
    'show_nodes': set(),
    'progress': 1.0,
    'interval': 3000,
    'frame_number': len(frames),
    'play_sound': False
})

print(f"\nTotal frames created: {len(frames)}")
print(f"At 600ms per frame: {len(frames) * 0.6:.1f} seconds = {len(frames) * 0.6 / 60:.1f} minutes")
print(f"\nExpected for first {slow_until} numbers:")
print(f"  {slow_until} numbers × {len(slow_frames)} frames = {slow_until * len(slow_frames)} frames")
print(f"  = {slow_until * len(slow_frames) * 0.6:.1f} seconds = {slow_until * len(slow_frames) * 0.6 / 60:.1f} minutes")

if len(frames) < 100:
    print("\n⚠️  WARNING: Very few frames generated!")
    print("This will make the animation very fast.")
    print("Check that slow_frames list is correct.")

node_last_used = {}

def animate(frame_idx):
    frame_data = frames[frame_idx]
    draw_organic_frame(ax, 
                      frame_data['current_n'],
                      frame_data['elements'],
                      frame_data['edges'],
                      all_levels,
                      frame_data['show_nodes'],
                      frame_data['progress'],
                      frame_data['frame_number'],
                      max_number,
                      frame_data.get('play_sound', False),
                      frame_data.get('is_fast', False))
    return []

anim = animation.FuncAnimation(fig, animate, frames=len(frames), 
                              interval=100, repeat=True, blit=False)

print("\nAnimation timing (target: 40 seconds):")
print(f"  Base interval: 100ms per frame")
print(f"  Numbers 1-10: {len(slow_frames)} frames = {len(slow_frames)*0.1:.1f}s each (smooth branch growth!)")
print(f"  Numbers 11-25: {len(medium_frames)} frames = {len(medium_frames)*0.1:.1f}s each")
print(f"  Numbers 26-60: 2 frames = 0.2s each")
print(f"  Numbers 61-300: 1 frame = 0.1s each (blazing!)")
print(f"\nTotal frames: {len(frames)}")
print(f"Total time: {len(frames)*0.1:.1f} seconds")
print(f"\nFor MP4/GIF export use: fps=2.5 (looks best)")
print("Actual frame interval is 100ms, but 2.5fps gives better pacing")

print("\n" + "="*70)
print("VERTICAL TREE with LIVING LEAF COLORS! 🍃")
print("="*70)
print("""
Tree grows UPWARD from root with better spacing:
  • Root (1) at BOTTOM CENTER
  • 15.0 units vertical spacing
  • Horizontal spacing proportional to NUMBER VALUES
  • Each level centered independently
  • Nodes reposition as tree grows

BEAUTIFUL LEAF AGING EFFECT:
  🟢 GREEN - Just used (vibrant, alive!)
  🟡 YELLOW - Aging (5-15 frames old)
  🟠 ORANGE - Getting old (15-30 frames)
  🔴 RED - Old (30-50 frames)
  🟤 BROWN - Dormant (50+ frames)
  
When a node is used again, it turns GREEN!
Watch leaves age and revitalize!

Branches: Brown, curve naturally
Active growth: Bright green
🔊 Sound: Wooden knock when each leaf appears!

Acceleration (MUCH slower start):
  • 1-50: VERY SLOW (600ms) - watch each leaf carefully
  • 51-150: MEDIUM (200ms) - building momentum
  • 151-300: FAST (50ms) - accelerating!
  • 301+: VERY FAST (10ms!) - explosive growth!
""")

print("\n" + "="*70)
print("SAVING OPTIONS:")
print("="*70)

print("\n1. Save animation as MP4 (silent):")
print("   anim.save('hasse_tree.mp4', writer='ffmpeg', fps=2.5, dpi=150, bitrate=1800)")

print("\n2. Generate audio track:")
print("   generate_audio_track(frames, 'hasse_audio.wav', base_interval=100)")

print("\n3. Combine video + audio with ffmpeg:")
print("   ffmpeg -i hasse_tree.mp4 -i hasse_audio.wav -c:v copy -c:a aac -shortest hasse_tree_with_audio.mp4")

print("\n4. Or save as GIF:")
print("   anim.save('hasse_tree.gif', writer='pillow', fps=2.5, dpi=100)")

print("\nffmpeg installation:")
print("  macOS:   brew install ffmpeg")
print("  Ubuntu:  sudo apt-get install ffmpeg")
print("  Windows: download from ffmpeg.org")

print("\nTo generate audio NOW, uncomment:")
print("# generate_audio_track(frames, 'hasse_audio.wav', base_interval=400)")

plt.show()

print("\nAnimation complete!")