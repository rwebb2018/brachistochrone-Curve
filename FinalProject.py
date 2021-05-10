"""
Edward Webb : ENAE380 Final Project

Implementing this program is as easy as running it. All of the functions are put into main() at the very bottom.
Note that commented beside (most) every equation is a reference to an equation in the formula guide. If the formula being used is unclear or hard to read, please seek the equation number listed in the reference guide for clarity.
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.integrate as integrate


# Intialize starting point to be (0,0)
x1 = 0 # Starting x
y1 = 0 # Starting y

# Initialize vertical distance to travelled to be 25m
y2 = -25 # Ending y

g = 9.81 # Acceleration due to gravity



def Brachistochrone(g,y2):
    """
    Traces the brachistochrone curve, Calculates the time taken for an object to travel 25m down along the curve (without friction), and calculates the distance travelled by the object.
    
    Parameters
    ----------
    r : float
        Radius of the rolling circle that traces the brachistochrone curve
        
    x2 : float
        Horizontal distance covered by the rolling circle within pi radians
        
    theta : numpy array
        Angles through which the circle will go through
        
    x : numpy array
        x values that are traced by the curve
        
    y : numpy array
        y values that are traced by the curve
        
    time : float
        Total time for the object to roll along the brachistochrone curve
        
    distance : float
        Total distance the rolling object covers on the brachistochrone curve
        
    
    Returns
    -------
    x : numpy array
        x values that are traced by the curve
        
    y : numpy array
        y values that are traced by the curve
        
    x2 : float
        Horizontal distance covered by the rolling circle within pi radians
        
    time : float
        Total time for the object to roll along the brachistochrone curve
    """
    # Solve for unkowns
    r = y2/2 # Calculate the radius of the circle that draws the cycloid curve (Eq 1)
    x2 = -3.14*r # Calculate how far the circle rolls in pi rotations (Eq 2)
    
    # Set up parametric equations
    theta = np.linspace(0, -np.pi, 100) # Set the angle through which to draw the cycloid
    
    x = r*(theta - np.sin(theta)) # Function to find the x values (Eq 3)
    y = r*(1 - np.cos(theta)) # Function to find the y values (Eq 4)
    
    
    # Calculate the time taken using the radius of the circle  
    time = round(np.pi*np.sqrt(r/-g), 3) # (Eq 5)
    
    
    # Calculate distance travelled
    distance = -4*r # (Eq 6)
    
    
    # Plot the function
    plt.plot(x,y)
    plt.title("Brachistochrone Curve")
    plt.xlim([0, x2])
    plt.ylim([y2, 0])
    plt.text(19, -4, "Time of descent: {0}s".format(time), fontsize=13, color='b', bbox={'facecolor': 'lightgrey'}) # Insert text box with time
    plt.text(17.6, -6.75, "Distance travelled: {0}m".format(distance), fontsize=13, color='b', bbox={'facecolor': 'lightgrey'}) # Insert text box with distance
    plt.savefig("Brachistochrone Curve", quality=95, dpi=250, transparent=True) # Save the plot to a file
    plt.show()
    
    
    return x,y,x2,time # Return values to be used in later function



def Linear(g,x,x2,y2):
    """
    Traces a linear curve, Calculates the time taken for an object to travel 25m down along the curve (without friction), and calculates the distance travelled by the object.
    
    Parameters
    ----------
    m : float
        Slope of the linear curve
        
    f_linear : numpy array
        The linear function
        
    time : float
        Total time for the object to roll along the linear curve
        
    distance : float
        Total distance the rolling object covers on the linear curve
        
    Returns
    -------
    f_linear : numpy array
        The linear function
     
    time : float
        Total time for the object to roll along the linear curve
    """
    # Calculate the function
    m = y2/x2 # Find slope of curve (Eq 7)
    
    f_linear = m*(x - x1) # Find curve (Eq 8)
    
    
    # Define the function that calculats the time of descent
    def Linear_Time_Func(x):
        # Function that is integrated to find the time
        return np.sqrt(1 + m**2) / np.sqrt(2*-g*m*x) # (Eq 9, without the integral)
    
    # Integrate the function to find time
    time = round(integrate.quad(Linear_Time_Func, 0, x2)[0],3) # (Eq 9)
    
    
    # Calculate distance travelled
    distance = round(np.sqrt(x2**2 + y2**2),1) # (Eq 10)
    

    # Plot the curve
    plt.plot(x,f_linear)
    plt.title("Linear Curve")
    plt.xlim([0, x2])
    plt.ylim([y2,0])
    plt.text(19, -4, "Time of descent: {0}s".format(time), fontsize=13, color='b', bbox={'facecolor': 'lightgrey'}) # Insert text box with time
    plt.text(17.6, -6.75, "Distance travelled: {0}m".format(distance), fontsize=13, color='b', bbox={'facecolor': 'lightgrey'}) # Insert text box with distance
    plt.savefig("Linear Curve", quality=95, dpi=250, transparent=True) # Save the plot to a file
    plt.show()
    
    return f_linear, time # Return values to be used in later function
  

    
def ParabolaUp(g,x,x2,y2):
    """
    Traces an upward facing parabolic curve, Calculates the time taken for an object to travel 25m down along the curve (without friction), and calculates the distance travelled by the object.
    
    Parameters
    ----------
    x_point : int
        Arbitrarily picked point to force the shape of the curve and because a parabola needs 3 given points to be drawn
        
    y_point : int
        Arbitrarily picked point to force the shape of the curve and because a parabola needs 3 given points to be drawn
        
    LHS_eqs : numpy array
        Left side of the parabolic equations when plugging in known values
        
    RHS_eqs : numpy array
        Right side of the parabolic equations when plugging in known values
        
    f_consts : list
        Holds A and B, the unknowns that need to be solved for to find the equation of a parabola
        
    A : float
        The first unknown to appear in the parabolic equation
        
    B : float
        The second unknown to appear in the parabolic equation
        
    f_parabolaup : numpy array
        The upward facing parabolic function
        
    time : float
        Total time for the object to roll along the upward facing parabolic curve
        
    distance : float
        Total distance the rolling object covers on the upward facing parabolic curve
        
    Returns
    -------
    f_parabolaup : numpy array
        The upward facing parabolic function
        
    time : float
        Total time for the object to roll along the upward facing parabolic curve
    """
    # Set an arbitary point to use in order to ensure the shape of the parabola
    x_point = 4
    y_point = -5
    
    # Set up equations used to solve for the unkowns in the parabola equation (A & B). (Eqs 12 & 13 put into arrays)
    LHS_eqs = np.array([[x2**2, x2], [x_point**2, x_point]]) # Left side of the two equations (Eqs 12 & 13, respectively)
    RHS_eqs = np.array([y2, y_point]) # Right side of the two equations (Eqs 12 & 13, respectively)

    f_consts = np.linalg.solve(LHS_eqs, RHS_eqs) # Solutions to the equations
    
    A = f_consts[0] # Pull out the first constant
    B = f_consts[1] # Pull out the second constant
    
    # Set up the equation of a parabola
    f_parabolaup = A*(x**2) + B*x # (Eq 11)
    
    
    # Define the function that calculates the time of descent
    def Parabola_time_func(x):
        # Function that is integrated to find the time
        return np.sqrt(1 + ((2*A)*x + B)**2) / np.sqrt(-2*g*(A*(x**2) + B*x)) # (Eq 14, without the integral)
    
    # Integrate to find the time
    time = round(integrate.quad(Parabola_time_func, 0, x2)[0],3) # (Eq 14)
    
    
    # Define the function that calculates the distance travelled
    def Parabola_length_func(x):
        # Function that is integrated to find the distance
        return np.sqrt(1 + (2*A*x + B)**2) # (Eq 15, without the integral)
    
    # Integrate to find the distance
    distance = round(integrate.quad(Parabola_length_func, 0, x2)[0],1) # (Eq 15)
    
    
    # Plot the function
    plt.plot(x,f_parabolaup)
    plt.title("Upward Parabolic Curve")
    plt.xlim([0,x2])
    plt.ylim([y2,0])
    plt.text(19, -4, "Time of descent: {0}s".format(time), fontsize=13, color='b', bbox={'facecolor': 'lightgrey'}) # Insert text box with time
    plt.text(17.6, -6.75, "Distance travelled: {0}m".format(distance), fontsize=13, color='b', bbox={'facecolor': 'lightgrey'}) # Insert text box with distance
    plt.savefig("Upward Parabola Curve", quality=95, dpi=250, transparent=True) # Save the plot to a file
    plt.show()
    
    return f_parabolaup,time # Return values to be used in later function
    
  
    
def ParabolaDown(g,x,x2,y2):
    """
    Traces a downward facing parabolic curve, Calculates the time taken for an object to travel 25m down along the curve (without friction), and calculates the distance travelled by the object.
    
    Parameters
    ----------
    x_point : int
        Arbitrarily picked point to force the shape of the curve and because a parabola needs 3 given points to be drawn
        
    y_point : int
        Arbitrarily picked point to force the shape of the curve and because a parabola needs 3 given points to be drawn
        
    LHS_eqs : numpy array
        Left side of the parabolic equations when plugging in known values
        
    RHS_eqs : numpy array
        Right side of the parabolic equations when plugging in known values
        
    f_consts : list
        Holds A and B, the unknowns that need to be solved for to find the equation of a parabola
        
    A : float
        The first unknown to appear in the parabolic equation
        
    B : float
        The second unknown to appear in the parabolic equation
        
    f_parabolaup : numpy array
        The upward facing parabolic function
        
    time : float
        Total time for the object to roll along the downward facing parabolic curve
        
    distance : float
        Total distance the rolling object covers on the downward facing parabolic curve
        
    Returns
    -------
    f_parabolaup : numpy array
        The upward facing parabolic function
        
    time : float
        Total time for the object to roll along the downward facing parabolic curve
    """
    # Set an arbitary point to use in order to ensure the shape of the parabola
    x_point = 15
    y_point = -5
    
    # Set up equations used to solve for the unkowns in the parabola equation (A & B). (Eqs 12 & 13 put into arrays)
    LHS_eqs = np.array([[x2**2, x2], [x_point**2, x_point]]) # Left side of the two equations (Eqs 12 & 13, respectively)
    RHS_eqs = np.array([y2, y_point]) # Right side of the two equations (Eqs 12 & 13, respectively)

    f_consts = np.linalg.solve(LHS_eqs, RHS_eqs) # Solutions to the equations
    
    A = f_consts[0] # Pull out the first constant
    B = f_consts[1] # Pull out the second constant
    
    # Set up the equation of a parabola
    f_paraboladown = A*(x**2) + B*x # (Eq 11)
    
    
    # Define the function that calculates the time of descent
    def Parabola_time_func(x):
        # Function that is integrated to find the time
        return np.sqrt(1 + ((2*A)*x + B)**2) / np.sqrt(-2*g*(A*(x**2) + B*x)) # (Eq 14, without the integral)
    
    # Integrate to find the time
    time = round(integrate.quad(Parabola_time_func, 0, x2)[0],3) # (Eq 14)
    
    
    # Define the function that calculates the distance travelled
    def Parabola_length_func(x):
        # Function that is integrated to find the distance
        return np.sqrt(1 + (2*A*x + B)**2) # (Eq 15, without the integral)
    
    # Integrate to find the distance
    distance = round(integrate.quad(Parabola_length_func, 0, x2)[0],1) # (Eq 15)
    
    
    # Plot the function
    plt.plot(x,f_paraboladown)
    plt.title("Downward Parabolic Curve")
    plt.xlim([0,x2])
    plt.ylim([y2,0])
    plt.text(3, -15, "Time of descent: {0}s".format(time), fontsize=13, color='b', bbox={'facecolor': 'lightgrey'}) # Insert text box with time
    plt.text(1.6, -17.75, "Distance travelled: {0}m".format(distance), fontsize=13, color='b', bbox={'facecolor': 'lightgrey'}) # Insert text box with distance
    plt.savefig("Downward Parabola Curve", quality=95, dpi=250, transparent=True) # Save the plot to a file
    plt.show()
    
    return f_paraboladown,time # Return values to be used in later function
    
    
    
def Circle(g,x2,y2):
    """
    Traces a circular curve, Calculates the time taken for an object to travel 25m down along the curve (without friction), and calculates the distance travelled by the object.
    
    Parameters
    ----------
    tri_distance : float
        distance between the start and end points
        
    r : float
        radius of the circle
        
    x_center : float
        x coordinate of the center of the circle
        
    y_center : float
        y coordinate of the center of the circle
        
    Eq1 : numpy array
        Circle equation given from plugging in the start point (0,0)
        
    Eq2 : numpy array
        Circle equation given from plugging in the end point (x2,y2)
        
    solns : list
        Solution to the two equations. Holds the x and y coordinates of the center
        
    t : numpy array
        array of theta values used in the parametric equations of a circle
        
    x : numpy array
        x values found from using t in the parametric equation of a circle
        
    y : numpy array
        y values found from using t in the parametric equations of a circle
        
    time : float
        Total time for the object to roll along the circular curve
        
    distance : float
        Total distance the rolling object covers on the downward facing parabolic curve
        
    Returns
    -------
    x : numpy array
        x values found from using t in the parametric equation of a circle
        
    y : numpy array
        y values found from using t in the parametric equations of a circle
        
    time : float
        Total time for the object to roll along the circular curve
    """
    # Find the radius of the circle
    tri_distance = np.sqrt(x2**2 + y2**2) # Find distance between the start and end points (Eq 16)
    r = np.sqrt((tri_distance**2)/2) # Solve for the radius using law of cosines (Eq 17)
    
    
    # Express the symbolic variables
    x_center, y_center = sp.symbols('x_center, y_center', real=True) 
    
    # Set up the equations to solve for the symbolic variables
    Eq1 = sp.Eq(x_center**2 + y_center**2, r**2) # First equation (from plugging (0,0) into the equation of a circle) (Eq 18)
    Eq2 = sp.Eq((x2-x_center)**2 + (y2 - y_center)**2, r**2) # Second equation (from plugging (x2,y2) into the equation of a cirlce) (Eq 19)
    
    solns = sp.nonlinsolve([Eq1, Eq2], [x_center, y_center]) # Solve the equations
    
    # Pull the values of the center from the solution set
    x_center = float(solns.args[1][0]) # x coordinate of the center of the circle
    y_center = float(solns.args[0][0]) # y coordinate of the center of the circle
    
    
    # Find x and y using the parametric equations of a circle
    t = np.linspace(0, np.pi, 100) # Establish the theta values to use
    x = x_center - r*np.cos(t) # Find the x values (Eq 20)
    y = y_center - r*np.sin(t) # Find the y values (Eq 21)
    
    # Define the function that calculates the time of descent
    def Circle_time_func(x):
        # Function that is integrated to find the time
        return np.sqrt(1 + ((x-x_center)/np.sqrt(r**2 - (x-x_center)**2))**2) / np.sqrt(-2*g*(y_center-np.sqrt(r**2 - (x-x_center)**2))) # (Eq 22, without integral)
    
    # Integrate to find the time
    time = round(integrate.quad(Circle_time_func, 0, x2)[0],3) # (Eq 22)
    
    
    # Calculate the distance travelled
    distance = round((np.pi/2)*r,1) # (Eq 23)
    
    # Plot the circle
    plt.plot(x,y)
    plt.title("Circular Curve")
    plt.xlim([0, x2])
    plt.ylim([y2, 0])
    plt.text(19, -4, "Time of descent: {0}s".format(time), fontsize=13, color='b', bbox={'facecolor': 'lightgrey'}) # Insert text box with time
    plt.text(17.6, -6.75, "Distance travelled: {0}m".format(distance), fontsize=13, color='b', bbox={'facecolor': 'lightgrey'}) # Insert text box with distance
    plt.savefig("Circular Curve", quality=95, dpi=250, transparent=True) # Save the plot to a file
    plt.show()
    
    
    return x,y,time # Return the x and y values calculated
    


def ComparePlots(x, Brach_data, Lin_data, Par_up_data, Par_down_data, Circle_data):
    """
    Plots every curve on the same plot to compare
    
    Parameters
    ----------
    Brach_x : numpy array
        x values from the Brachistochrone function
        
    Brach_y : numpy array
        y values from the Brachistochrone function
        
    Brach_time : float
        time from the Brachistochrone function
        
    Lin_y : numpy array
        y values from the linear function
        
    Lin_time : float
        time from the linear function
        
    Par_up_y : numpy array
        y values from the upward parabola function
        
    Par_up_time : float
        time from the upward parabola function
        
    Par_down_y : numpy array
        y values from the downward parabola function
        
    Par_down_time : float
        time from the downward parabola function
        
    Circle_x : numpy array
        x values from the circle function
        
    Circle_y : numpy array
        y values from the circle function
        
    Circle_time : float
        time from the circle function
    """
    # Start Brachistochrone
    Brach_x = Brach_data[0] # Pull x values
    Brach_y = Brach_data[1] # Pull y values
    Brach_time = Brach_data[3] # Pull time
    
    # Plot the brachistochrone curve
    plt.plot(Brach_x, Brach_y, label='Brachistochrone: {0}s'.format(Brach_time), color='cyan')
    
    
    # Start Linear
    Lin_y = Lin_data[0] # Pull y values
    Lin_time = Lin_data[1] # Pull time
    
    # Plot the linear curve
    plt.plot(x,Lin_y, label='Linear: {0}s'.format(Lin_time), color='orange')
    
    
    # Start Parabola Up
    Par_up_y = Par_up_data[0] # Pull y values
    Par_up_time = Par_up_data[1] # Pull time
    
    # Plot upwared parabola curve
    plt.plot(x, Par_up_y, label='Upward Parabola: {0}s'.format(Par_up_time), color='green')
    
    
    # Start Parabola Down
    Par_down_y = Par_down_data[0] # Pull y values
    Par_down_time = Par_down_data[1] # Pull time
    
    # Plot downward parabola curve
    plt.plot(x, Par_down_y, label='Downward Parabola: {0}s'.format(Par_down_time), color='red')
    
    
    # Start circle
    Circle_x = Circle_data[0] # Pull x values
    Circle_y = Circle_data[1] # Pull y values
    Circle_time = Circle_data[2] # Pull time
    
    # Plot circular curve
    plt.plot(Circle_x,Circle_y, label='Circle: {0}s'.format(Circle_time), color='purple')
    
    
    # Label the plot
    plt.title("Comparison Plot")
    plt.xlim([0, x[len(x) -1]])
    plt.legend()
    plt.savefig("Comparison Plot", quality=95, dpi=250, transparent=True) # Save the plot to a file
    plt.show()
    
    
    
# Main function to run all of the function. (I apologize if this is implemented incorrectly, the software I use does not require this to run things.)
def main():  
    # Run the brachistochrone function
    brachistochrone_data = Brachistochrone(g,y2)

    # Get values from running the brachistochrone function
    x2 = brachistochrone_data[2] # Get how far the curves will go 
    x = np.linspace(0, x2, 100) # Make an array of x values to use for the functions that will need it


    # Run the rest of the curve functions
    linear_data = Linear(g,x,x2,y2)
    parabolaUp_data = ParabolaUp(g,x,x2,y2)
    parabolaDown_data = ParabolaDown(g,x,x2,y2)
    circle_data = Circle(g,x2,y2)


    # Run the comparison
    ComparePlots(x, brachistochrone_data, linear_data, parabolaUp_data, parabolaDown_data, circle_data)

# Run main
if __name__ == "__main__":
    main()
    