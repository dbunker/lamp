d0(o2).
d1(o3).
d2(o5).
d2(o3).
d4(o3).
d4(o4).
d5(o4).
d0(o3).
d0(o4).

d3(X) :- d4(X), d0(X).
d5(X) :- d4(X), d2(X).
d3(X) :- d4(X), d1(X).
d2(X) :- d4(X), d5(X).
d4(X) :- d5(X), d3(X).
d3(X) :- d2(X), d0(X).
d0(X) :- d2(X), d3(X).
d0(X) :- d2(X), d1(X).
d3(X) :- d4(X), d2(X).
d5(X) :- d2(X), d3(X).
d4(X) :- d1(X), d0(X).