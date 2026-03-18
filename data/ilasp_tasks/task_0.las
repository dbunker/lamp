#constant(obj, o2).
#constant(obj, o4).
#constant(obj, o5).
#constant(obj, o6).
#constant(obj, o7).
#constant(obj, o9).

#modeh(d1(var(obj))).
#modeh(d3(var(obj))).
#modeh(d8(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).
#modeb(1, d6(var(obj))).
#modeb(1, d6(var(obj)), (negative)).
#modeb(1, d7(var(obj))).
#modeb(1, d7(var(obj)), (negative)).
#modeb(1, d8(var(obj))).
#modeb(1, d8(var(obj)), (negative)).

#pos(eg1, {
  d1(o4), d1(o9), d3(o2), d8(o2)
}, {
  d1(o2), d1(o5), d3(o4), d3(o5), d3(o6), d3(o7), d3(o9), d8(o4), d8(o5), d8(o6), d8(o7), d8(o9)
}, {
  d0(o4).
  d1(o6).
  d1(o7).
  d4(o4).
  d4(o9).
  d5(o4).
  d6(o2).
  d7(o2).
  d7(o5).
}).