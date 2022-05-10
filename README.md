# A-PINN

These are the code attachments of paper 'A-PINN: Auxiliary physics informed neural networks for forward and inverse problems of nonlinear integro-differential equations'. This paper is published at Journal of Computational Physics. In this paper, we presented the auxiliary physics informed neural network (A-PINN) framework for solving the forward and inverse problems of nonlinear IDEs. In the proposed A-PINN, a multi-output neural network is configured to simultaneously represent the primary variables and integrals in the governing equations. By pursuing automatic differentiation of the auxiliary outputs in lieu of the integral operators in IDEs, we bypass the limitation of neural networks in dealing with integral manipulation. As integral discretization is avoided, A-PINN doesn’t suffer from discretization and truncation errors for forward and inverse solution of IDEs. Because of being devoid of fixed grids or nodes, A-PINN is a mesh-free method that can calculate/predict the solution at any point in the equation domain without interpolation. 


For more information, please refer to the following:

Yuan, L., Ni, Y. Q., Deng, X. Y., & Hao, S. (2022). A-PINN: Auxiliary physics informed neural networks for forward and inverse problems of nonlinear integro-differential equations. Journal of Computational Physics, 111260. https://www.sciencedirect.com/science/article/pii/S0021999122003229

Citation: 

@article{yuan2022pinn,  
  title={A-PINN: Auxiliary physics informed neural networks for forward and inverse problems of nonlinear integro-differential equations},  
  author={Yuan, Lei and Ni, Yi-Qing and Deng, Xiang-Yun and Hao, Shuo},  
  journal={Journal of Computational Physics},  
  pages={111260},  
  year={2022},  
  publisher={Elsevier}  
}


