d1(o8).
d0(o9).
d2(o8).
d4(o2).
d3(o5).
d1(o3).

d2(X) :- d0(X), d5(X).
d4(X) :- d0(X), d3(X).
d1(X) :- d3(X), d4(X).
d0(X) :- d2(X), d1(X).
d0(X) :- d5(X), d4(X).
d5(X) :- d2(X), d3(X).
d0(X) :- d3(X), d4(X).
d2(X) :- d1(X), d4(X).
d2(X) :- d3(X), d5(X).
d2(X) :- d0(X), d5(X).
d1(X) :- d0(X), d5(X).