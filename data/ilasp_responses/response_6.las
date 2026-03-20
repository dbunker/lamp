%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.008s
%% Hypothesis Space Generation             : 2.351s
%% Conflict analysis                       : 45.29s
%%   - Positive Examples                   : 45.29s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0.002s
%% Hypothesis Search                       : 15.681s
%% Total                                   : 64.752s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d7(V1) :- d1(V1).
d6(V1) :- d7(V1).
d5(V1) :- d1(V1).
d4(V1) :- d8(V1).
d2(V1) :- d3(V1).
d3(V1) :- d4(V1).

