d5(o3).
d2(o3).
d5(o5).
d4(o3).
d5(o4).

d1(X) :- d5(X), d2(X).
d0(X) :- d5(X), d4(X).
d3(X) :- d5(X), d2(X).