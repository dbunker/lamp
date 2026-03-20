%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing                          : 0.007s
%% Hypothesis Space Generation             : 0.066s
%% Conflict analysis                       : 0.125s
%%   - Positive Examples                   : 0.125s
%% Counterexample search                   : 0.002s
%%   - CDOEs                               : 0s
%%   - CDPIs                               : 0s
%% Hypothesis Search                       : 0.058s
%% Total                                   : 0.266s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d3(V1) :- d1(V1).
d1(V1) :- d0(V1).

