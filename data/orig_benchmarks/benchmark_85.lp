d5(o8).
d1(o8).
d0(o5).
d3(o8).
d2(o0).

d1(X) :- d4(X), d0(X), d2(X).
d5(X) :- d2(X), d1(X), d3(X).
d2(X) :- d0(X), d5(X), d3(X).
d2(X) :- d1(X), d0(X), d3(X).
d3(X) :- d4(X), d1(X), d0(X).
d0(X) :- d4(X), d5(X), d3(X).
d1(X) :- d4(X), d2(X), d3(X).
d4(X) :- d1(X), d5(X), d3(X).
d5(X) :- d4(X), d0(X), d2(X).
d5(X) :- d1(X), d0(X), d3(X).
d4(X) :- d2(X), d0(X), d3(X).